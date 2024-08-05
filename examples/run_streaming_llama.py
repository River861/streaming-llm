import warnings

warnings.filterwarnings("ignore")

import torch
import argparse
import json
import os
import time
import re
import sys

from tqdm import tqdm
from streaming_llm.utils import load, download_url, load_jsonl
from streaming_llm.enable_streaming_llm import enable_streaming_llm


@torch.no_grad()
def greedy_generate(model, tokenizer, input_ids, past_key_values, max_gen_len):
    # prefill阶段
    outputs = model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )
    # !!! past_key_values的大小是(40, 2), 40是因为model里一共有40个attention层, 2分别指key和value张量
    past_key_values = outputs.past_key_values
    # logits指模型最后一层的输出(还没有应用激活函数)
    # output.logits形状: (batch_size, seq_len, vocab_size), 表示每个时间步(token)时每个词的分数
    # 所以这里就是取最后一个时间步, 找分数最高的词作为输出, unsqueeze(1)的作用是把最后的[batch_size]张量变成[batch_size, 1]
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    generated_ids = [pred_token_idx.item()]  # item()就是把值从张量里取出来而已, 这里先放进了prefill阶段生成的first token
    pos = 0
    # decode阶段
    for _ in range(max_gen_len - 1):  # 每轮迭代
        print('\033[94m' + f"\n[greedy_generate]: past_key_values.size={(len(past_key_values), len(past_key_values[0]), (past_key_values[0][0].size(), past_key_values[0][1].size())) if past_key_values is not None else None}" + '\033[0m')
        outputs = model(
            input_ids=pred_token_idx,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values  # 每轮迭代会增加past_key_values
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)  # 和prefill一样生成token(ID)
        generated_ids.append(pred_token_idx.item())
        generated_text = (
            tokenizer.decode(  # 用分词器把生成的token IDs重新映射回文本
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
                spaces_between_special_tokens=False,
            )
            .strip()
            .split(" ")
        )

        now = len(generated_text) - 1
        if now > pos:
            print(" ".join(generated_text[pos:now]), end=" ", flush=True)  # 输出文本, flush表示立即打印
            pos = now

        if pred_token_idx == tokenizer.eos_token_id:  # 生成eos时提前终止
            break
    print(" ".join(generated_text[pos:]), flush=True)
    return past_key_values


@torch.no_grad()
def streaming_inference(model, tokenizer, prompts, kv_cache=None, max_gen_len=1000):  # 开始跑workloads
    past_key_values = None  # !! past_key_values是跨prompt维护的, 这样才是实现reuse
    for idx, prompt in enumerate(prompts):
        print('\033[94m' + f"\n[streaming_inference]: past_key_values.size={(len(past_key_values), len(past_key_values[0]), (past_key_values[0][0].size(), past_key_values[0][1].size())) if past_key_values is not None else None}" + '\033[0m')
        prompt = "USER: " + prompt + "\n\nASSISTANT: "
        print("\n" + prompt, end="")
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids  # 分词, 生成每个词的word embedding向量, input_ids指每个token(在词汇表中)被映射到的唯一整数id
        input_ids = input_ids.to(model.device)  # 把input_ids传到model所在的设备上(比如GPU)
        seq_len = input_ids.shape[1]  # input_ids例子:tensor([[ 101, 7592, 1010, 2129, 2024, 2017,  102]])
        if kv_cache is not None:
            space_needed = seq_len + max_gen_len  # 需要的KV-Cache space是prompt长+max生成长度
            past_key_values = kv_cache.evict_for_space(past_key_values, space_needed)  # TODO: 跨prompt的话, 这个eviction是不是有问题?

        past_key_values = greedy_generate(
            model, tokenizer, input_ids, past_key_values, max_gen_len=max_gen_len
        )


def main(args):
    model_name_or_path = args.model_name_or_path
    model, tokenizer = load(model_name_or_path)
    test_filepath = os.path.join(args.data_root, "mt_bench.jsonl")
    print(f"Loading data from {test_filepath} ...")

    if not os.path.exists(test_filepath):
        download_url(
            "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl",
            args.data_root,
        )
        os.rename(os.path.join(args.data_root, "question.jsonl"), test_filepath)

    list_data = load_jsonl(test_filepath)
    prompts = []
    for sample in list_data:
        prompts += sample["turns"]

    if args.enable_streaming:  # 这里的kv_cache是KV tensors的一个manager而已, 真正的KV tensors是past_key_values
        kv_cache = enable_streaming_llm(
            model, start_size=args.start_size, recent_size=args.recent_size
        )
    else:
        kv_cache = None

    streaming_inference(
        model,
        tokenizer,
        prompts,
        kv_cache,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="lmsys/vicuna-13b-v1.3"
    )
    parser.add_argument("--data_root", type=str, default="data/")
    parser.add_argument("--enable_streaming", action="store_true")
    parser.add_argument("--start_size", type=int, default=4)
    parser.add_argument("--recent_size", type=int, default=2000)
    args = parser.parse_args()

    main(args)
