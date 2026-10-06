import os

# BitNest model registry: env MODEL=<key> -> (HF model id, architecture, rotation path, log tag).
# Override the model location with INPUT_MODEL=<local dir or HF id> and the rotation with ROT=<path/to/R.bin>.

REGISTRY = {
    "llama2":      dict(input_model="meta-llama/Llama-2-7b-hf",        arch="llama", rot="outputs/llama2_w4a8/R.bin",      tag="LLAMA2"),
    "llama3":      dict(input_model="meta-llama/Meta-Llama-3-8B",      arch="llama", rot="outputs/llama3_w4a8/R.bin",      tag="LLAMA3"),
    "qwen2":       dict(input_model="Qwen/Qwen2-7B",                   arch="qwen2", rot="outputs/qwen2_w4a8/R.bin",       tag="QWEN2"),
    "qwen2.5":     dict(input_model="Qwen/Qwen2.5-7B",                 arch="qwen2", rot="outputs/qwen2.5_w4a8/R.bin",     tag="QWEN2.5"),
    # long-context model (MHA, linear RoPE scaling x8, 32K)
    "llama2_32k":  dict(input_model="togethercomputer/LLaMA-2-7B-32K", arch="llama", rot="outputs/llama2_32k_w4a8/R.bin",  tag="LLAMA2_32K"),
    # 3B edge models (tie_word_embeddings; lm_head is nested afterwards with bitnest/nest_lm_head.py)
    "llama3.2_3b": dict(input_model="meta-llama/Llama-3.2-3B",         arch="llama", rot="outputs/llama3.2_3b_w4a8/R.bin", tag="LLAMA3.2_3B"),
    "qwen2.5_3b":  dict(input_model="Qwen/Qwen2.5-3B",                 arch="qwen2", rot="outputs/qwen2.5_3b_w4a8/R.bin",  tag="QWEN2.5_3B"),
}


def get_entry(name=None):
    name = name or os.environ.get("MODEL", "llama2")
    if name not in REGISTRY:
        raise KeyError(f"MODEL={name} is not registered; available: {list(REGISTRY)}")
    ent = dict(REGISTRY[name], name=name)
    if os.environ.get("INPUT_MODEL"):
        ent["input_model"] = os.environ["INPUT_MODEL"]
    return ent


def get_model_class(arch):
    if arch == "llama":
        from eval_utils.modeling_llama import LlamaForCausalLM
        return LlamaForCausalLM
    if arch == "qwen2":
        from eval_utils.modeling_qwen2 import Qwen2ForCausalLM
        return Qwen2ForCausalLM
    raise KeyError(f"unknown architecture {arch}")
