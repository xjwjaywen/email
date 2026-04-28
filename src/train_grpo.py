"""
train_grpo.py — GRPO 训练 Qwen2.5-7B + LoRA on MATH (R1-Zero 路线)

干啥的:
    用 GRPO (DeepSeek-R1 同款 RL 算法) 训练数学推理能力.
    每道题让模型生成 N (=8) 个不同答案, 答对的奖励 +1 答错 0,
    用 "组内相对优势" 做 policy gradient 更新, 跳过 SFT 直接 RL.

输入:  qwedsacf/competition_math (在线下载, 同 SFT)
输出:  outputs/v02_grpo/  (LoRA adapter)

跑法 (避开别人占用的 GPU):
    CUDA_VISIBLE_DEVICES=3 python -m src.train_grpo --out outputs/v02_grpo --max_samples 1000

为什么走这条路 (v0.1 SFT failure 的反思):
    SFT 让 7B 模仿 MATH 数据集的紧凑解答风格, 导致 response 长度
    崩到原来 70% (500→352 chars), 算术失误增多, 净效果 -1pp.
    R1-Zero 论文证明: 已经预训练好的 base (Qwen2.5-7B) 不需要 SFT 教格式,
    直接用 verifiable reward (\\boxed{} 答案对错) 做 RL 即可, 而且没有
    "数据风格污染" 这个 SFT 特有的失败模式.

预期: 比 baseline 41.4% 涨 5-15pp, 训练时长 1000 题 ~4-8 小时.
"""

import argparse
import re
from pathlib import Path

import torch
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from unsloth import FastLanguageModel


SYSTEM_PROMPT = (
    "You are a math problem solver. Solve the problem step by step, "
    "showing your reasoning. Put your final answer inside \\boxed{}."
)

# 同 eval.py 的抠答案逻辑, 训练-评测一致, 避免 reward / accuracy 不对齐
BOXED_PATTERN = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")


def extract_boxed(text: str) -> str | None:
    matches = BOXED_PATTERN.findall(text)
    return matches[-1].strip() if matches else None


def normalize(s: str) -> str:
    """轻量归一化, 让 'p - q' 和 'p-q' 算相同"""
    return s.replace(" ", "").lower() if s else ""


# ─── reward 函数 ─────────────────────────────────────────────────────
# TRL GRPOTrainer 调用约定:
#   reward_func(prompts: List[str], completions: List[str], **kwargs) -> List[float]
#   kwargs 里会有 dataset 中的所有非 prompt 列, 我们的 dataset 加了 'gold' 列.

def correctness_reward(prompts, completions, gold, **kwargs):
    """主奖励: \\boxed{X} 抠出来 == gold 给 1.0, 否则 0.0
    这是 verifiable reward — 不要 LLM judge, 不要相似度, 只要程序化对错.
    """
    rewards = []
    for completion, g in zip(completions, gold):
        pred = extract_boxed(completion)
        if pred is None or not g:
            rewards.append(0.0)
        else:
            rewards.append(1.0 if normalize(pred) == normalize(g) else 0.0)
    return rewards


def format_reward(prompts, completions, **kwargs):
    """次要奖励: 至少出现一次 \\boxed{} 给 0.1, 鼓励"输出格式正确".
    比起 correctness reward 小一个量级, 不会主导训练.
    """
    return [0.1 if extract_boxed(c) else 0.0 for c in completions]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True,
                        help="输出目录, GRPO LoRA 存这里")
    parser.add_argument("--model", default="unsloth/Qwen2.5-7B-Instruct",
                        help="base 模型 (Unsloth 4-bit)")
    parser.add_argument("--max_samples", type=int, default=1000,
                        help="只用 N 道题 (GRPO 慢, 1000 题 ~4-8 小时)")
    parser.add_argument("--filter_levels", default="Level 3,Level 4",
                        help="只训这些难度 (逗号分隔). 太简单 = 8/8 全对无信号; "
                             "太难 = 8/8 全错无信号. 中等难度学习效率最高. "
                             "传 '' 关闭过滤 (用全部难度)")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--max_prompt_length", type=int, default=512,
                        help="prompt 最长 token 数, MATH 题目一般 <300")
    parser.add_argument("--max_completion_length", type=int, default=1536,
                        help="生成最长 token, 数学解答可能很长. 1024 在冒烟测试中"
                             "对部分难题会截断, 1536 给硬题足够推理空间")
    parser.add_argument("--num_generations", type=int, default=8,
                        help="每道题生成几个候选 (R1 默认 8)")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-6,
                        help="RL 用很小的 lr, SFT 是 2e-4, GRPO 一般 1e-6 ~ 1e-5")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="每张卡每步几道 prompt (每道再生成 N 个), 默认 1")
    parser.add_argument("--grad_accum", type=int, default=8,
                        help="梯度累积, eff prompt batch = batch_size × grad_accum. "
                             "TRL 0.18+ 要求 batch_size × grad_accum 能整除 num_generations")
    parser.add_argument("--epochs", type=int, default=1,
                        help="GRPO 一般 1 epoch 就够, 多了容易崩")
    parser.add_argument("--beta", type=float, default=0.04,
                        help="KL 约束权重, 防止偏离 base 太远 (R1 默认 0.04)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="生成 sampling 温度, 1.0 给足够多样性")
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=100)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # ─── 1. 加载 base 模型 + 4-bit 量化 ──────────────────────────────────
    print(f"━━━ 加载 base model: {args.model} ━━━")
    model, tokenizer = FastLanguageModel.from_pretrained(
        args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        device_map={"": 0},
    )

    # ─── 2. 贴 LoRA 补丁 ────────────────────────────────────────────────
    # 注: GRPOTrainer 会自动用 PEFT 的 adapter-disabling 技巧算 KL 参考分布,
    # 不需要再加载一份 reference 模型, 节省 ~5GB 显存
    print(f"━━━ 加 LoRA 补丁 (r={args.lora_r}) ━━━")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.0,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
    )

    # eos_token 防御 (同 train.py, Qwen 包装有时 eos_token 是 None)
    QWEN_EOS = "<|im_end|>"
    if tokenizer.eos_token is None or tokenizer.eos_token == "<EOS_TOKEN>":
        tokenizer.eos_token = QWEN_EOS
    print(f"  tokenizer.eos_token = {tokenizer.eos_token!r}")

    # ─── 3. 加载数据 — GRPO 只要 prompt + gold, 不要 solution ─────────────
    print(f"━━━ 加载 MATH 训练集 ━━━")
    ds = load_dataset("qwedsacf/competition_math", split="train")
    print(f"  全集大小: {len(ds)}")

    # 难度过滤: GRPO 在 8/8 全对或 8/8 全错的批次上无信号 (废算力).
    # 只留中等难度可以让 ~每个 batch 都有学习信号.
    if args.filter_levels:
        wanted = set(s.strip() for s in args.filter_levels.split(","))
        ds = ds.filter(lambda ex: ex.get("level", "") in wanted)
        print(f"  按难度过滤 ({wanted}): {len(ds)}")

    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    print(f"  截取后: {len(ds)}")

    def format_example(ex):
        # 拼 prompt (包含 system + user, 末尾 assistant 引导符)
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": ex["problem"].strip()},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        gold = extract_boxed(ex["solution"])
        return {"prompt": prompt, "gold": gold or ""}

    ds = ds.map(format_example, remove_columns=ds.column_names)
    # 过滤抠不出 gold 的 (无法验证, 留着也没用还会拉低 reward)
    before = len(ds)
    ds = ds.filter(lambda ex: ex["gold"] != "")
    print(f"  过滤后: {len(ds)} (丢弃了 {before - len(ds)} 条无 gold 的)")

    # ─── 4. GRPO 配置 ───────────────────────────────────────────────────
    grpo_config = GRPOConfig(
        output_dir=str(args.out),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        beta=args.beta,
        temperature=args.temperature,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        save_total_limit=2,
        report_to="none",
        seed=42,
        remove_unused_columns=False,   # 关键: 保留 'gold' 列给 reward func 用
    )

    # ─── 5. 训练 ────────────────────────────────────────────────────────
    eff_prompt_batch = args.batch_size * args.grad_accum
    eff_seq_batch = eff_prompt_batch * args.num_generations
    total_updates = (len(ds) * args.epochs) // eff_prompt_batch
    print(f"━━━ 训练配置 ━━━")
    print(f"  样本数:               {len(ds)}")
    print(f"  每题生成:             {args.num_generations}")
    print(f"  prompt 累积 batch:    {eff_prompt_batch}")
    print(f"  序列 batch (含生成):  {eff_seq_batch}")
    print(f"  预计 updates:         ~{total_updates}")
    print(f"  lr:                   {args.lr}")
    print(f"  beta (KL):            {args.beta}")

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=ds,
        reward_funcs=[correctness_reward, format_reward],
        processing_class=tokenizer,
    )

    print(f"\n━━━ 开始 GRPO 训练 (预计 4-8 小时) ━━━\n")
    trainer.train()

    # ─── 6. 保存 ────────────────────────────────────────────────────────
    print(f"\n━━━ 保存 LoRA → {args.out} ━━━")
    model.save_pretrained(str(args.out))
    tokenizer.save_pretrained(str(args.out))

    print(f"\n✅ GRPO 训练完成")
    print(f"   下一步: CUDA_VISIBLE_DEVICES=3 python -m src.eval --num_problems 500 \\")
    print(f"           --lora {args.out} --out outputs/eval_grpo_500.json")


if __name__ == "__main__":
    main()
