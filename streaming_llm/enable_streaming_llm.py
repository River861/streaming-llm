from streaming_llm.kv_cache import StartRecentKVCache


def enable_streaming_llm(model, start_size, recent_size):
    print(f"model type: {model.config.model_type}")
    if "llama" in model.config.model_type:
        k_seq_dim = v_seq_dim = 2
        from streaming_llm.pos_shift.modify_llama import (
            enable_llama_pos_shift_attention,
        )

        enable_llama_pos_shift_attention(model)
    elif "mpt" in model.config.model_type:
        v_seq_dim = 2
        k_seq_dim = 3
    elif "gpt_neox" in model.config.model_type:
        k_seq_dim = v_seq_dim = 2
        from streaming_llm.pos_shift.modify_gpt_neox import (
            enable_gpt_neox_pos_shift_attention,
        )

        enable_gpt_neox_pos_shift_attention(model)
    elif "falcon" in model.config.model_type:
        v_seq_dim = 1
        k_seq_dim = 1
        from streaming_llm.pos_shift.modify_falcon import (
            enable_falcon_pos_shift_attention,
        )

        enable_falcon_pos_shift_attention(model)
    else:
        raise ValueError(f"got {model.config.model_type}")
    kv_cache = StartRecentKVCache(  # 这个kv_cache貌似是past_key_value的处理类, 类似manager
        start_size=start_size,   # attention sink数量
        recent_size=recent_size, # KV-Cache windows大小
        k_seq_dim=k_seq_dim,  # 这个貌似是指key和value向量矩阵分别在第几维是seq_len维, 这一维是可以切分的
        v_seq_dim=v_seq_dim,
    )
    return kv_cache
