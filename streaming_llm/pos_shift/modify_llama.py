import math
from typing import Optional, Tuple

import torch
from torch import nn
import torch.utils.checkpoint

import torch.nn.functional as F

from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    rotate_half,
    apply_rotary_pos_emb,
    repeat_kv,
)
import types

__all__ = ["enable_llama_pos_shift_attention"]


def apply_rotary_pos_emb_single(x, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


def llama_pos_shift_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    if self.config.pretraining_tp > 1:  # Tensor Parallelism
        # num_heads是attention header的数量，head_dim是q/k/v=W*x后的列数
        # x: (batch_size, seq_length(tokens), input_dim), W: (head_dim, input_dim) => Wx: (batch_size, seq_length, head_dim)
        # 所以这里计算的是KV tensors的投影矩阵(W)的切分，要将W划分到不同GPU进行张量并行
        key_value_slicing = (
            self.num_key_value_heads * self.head_dim
        ) // self.config.pretraining_tp
        # 利用算好的slicing对投影矩阵W进行按行(dim=0)切分(也就是对W的head_dim切分)
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)
        # 切分完模型的计算变成：每个切片(在一个GPU上)进行linear计算，最后最后再拼接起来(也就是对Wx的head_dim拼接)
        query_states = [
            F.linear(hidden_states, query_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [
            F.linear(hidden_states, key_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [
            F.linear(hidden_states, value_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        value_states = torch.cat(value_states, dim=-1)

    else:
        # 如果没有多GPU，就直接算q/k/v=Wx即可
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    # q/k/v向量: (batch_size, num_heads, q_len, head_dim)
    query_states = query_states.view(  # view就是reshape
        bsz, q_len, self.num_heads, self.head_dim
    ).transpose(1, 2)  # transpose是交换两个维度
    key_states = key_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]  # seq_len就是q_len
    # 处理q: 进行旋转位置编码
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]  # 这里加过去KV的q_len，像是在算当前token的position
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)  # 根据当前token的position计算位置编码
    ### Shift Pos: query pos is min(cache_size, idx)
    # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    # print('\033[94m' + f"Learn: position_ids={position_ids} kv_seq_len={kv_seq_len}" + '\033[0m')
    query_states = apply_rotary_pos_emb_single(query_states, cos, sin, position_ids)  # position_ids就是0,1,2,3,...!!! len(position_ids)==kv_seq_len!!!
    ###

    # 处理k,v: 先reuse KV-Cache再进行旋转位置编码
    if past_key_value is not None:
        # reuse k, v, self_attention
        # !!! KV tensors: (batch_size=1, num_heads=40, q_len=38,39,40,..., head_dim=128)
        key_states = torch.cat([past_key_value[0], key_states], dim=2)  # 在q_len维度拼接，其实就是将KV-Cache续上
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None  # 这个past_key_value就是在迭代过程中一直维护的KV-Cache

    ### Shift Pos: key pos is the pos in cache
    key_position_ids = torch.arange(kv_seq_len, device=position_ids.device).unsqueeze(0)  # TODO 这个key_position_ids其实就等于position_ids?
    key_states = apply_rotary_pos_emb_single(key_states, cos, sin, key_position_ids)
    print('\033[94m' + f"\n[llama_pos_shift_attention_forward]: position_ids={position_ids} \nkey_position_ids={key_position_ids}" + '\033[0m')
    ###

    # repeat k/v heads if n_kv_heads < n_heads
    # repeat_kv的作用是让KV tensors在batch_size那一维度进行复制, 复制的数目为num_key_value_groups(就是attention header的数目)
    # 目的是为了适应多注意力头的计算, 让每个注意力头使用相同的KV tensors
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # 计算attention score: QK^T
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
        self.head_dim  # 这个就是公式里的d
    )

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):  # 查一下q_len和kv_seq_len啥时候不一样: 在self-attention上二者是一样长的, 在跨注意力机制上才不一样长(比如机器翻译等)
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:  # mask矩阵就是用来让每个token只看到前面tokens的attention
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask  # 为啥是加法? 因为mask矩阵中0表示不屏蔽, -inf表示屏蔽

    # 计算output: softmax(score)*V
    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()  # contiguous是将tensor中的数据变得内存连续(因为转置/切片会让数据不连续)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)  # TODO: hidden_size应该就等于head_dim

    # 多GPU情形处理ouput的张量并行
    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(
            self.hidden_size // self.config.pretraining_tp, dim=2
        )
        o_proj_slices = self.o_proj.weight.split(
            self.hidden_size // self.config.pretraining_tp, dim=1
        )
        attn_output = sum(
            [
                F.linear(attn_output[i], o_proj_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
        )
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    # print('\033[94m' + f"greedy_generate: past_key_value type={type(past_key_value)} len={len(past_key_value) if past_key_value is not None else None}" + '\033[0m')
    return attn_output, attn_weights, past_key_value  # (attention层输出, attention score, 前面tokens的KV tensors)


def enable_llama_pos_shift_attention(model):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            enable_llama_pos_shift_attention(
                module,
            )

        if isinstance(module, LlamaAttention):
            # print('\033[94m' + f"attention layer!!" + '\033[0m')  # 测出来总共有40个attention layers
            model._modules[name].forward = types.MethodType(
                llama_pos_shift_attention_forward, model._modules[name]
            )
