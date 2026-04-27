"""
train.py — SFT 训练 Qwen2.5-7B + LoRA on MATH

干啥的:
    读 data_prep.py 生成的 jsonl, 用 Unsloth + TRL 微调 Qwen2.5-7B,
    只训 LoRA 补丁 (~16M 参数), 不动 base 模型 (70 亿参数).
    用 train_on_responses_only 让 loss 只算 assistant 部分.
    训完保存 LoRA 文件 (~50MB) 到 outputs/.

输入:  data/sft_train.jsonl   (来自 data_prep.py, 12500 条)
输出:  outputs/v01_sft/       (LoRA adapter + tokenizer)

跑法 (避开别人占用的 GPU):
    CUDA_VISIBLE_DEVICES=3 python -m src.train --data data/sft_train.jsonl --out outputs/v01_sft

预期看到:
    loss 数字从 ~1.5 一路降到 ~0.5-0.8 左右, 这就是训练成功的标志.
    显存占用 ~16-20 GB.
    训练时长: 12500 题 × 3 epoch / (2*8) eff batch ≈ 2300 步, 约 2-4 小时.
"""

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True,
                        help="SFT 数据 jsonl 文件 (来自 data_prep.py)")
    parser.add_argument("--out", type=Path, required=True,
                        help="输出目录, 训完的 LoRA 存这里")
    parser.add_argument("--model", default="unsloth/Qwen2.5-7B-Instruct",
                        help="base 模型 (Unsloth 4-bit 版)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="数据过几遍, MATH 任务 3 epoch 比较稳")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="学习率, LoRA 用 1e-4 ~ 5e-4 都行")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA 补丁宽度, 越大学得多但越慢")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="单条数据最长 token 数")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="一次喂几条")
    parser.add_argument("--grad_accum", type=int, default=8,
                        help="梯度累积, eff batch = batch_size × grad_accum")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="多少步打一次 loss")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="多少步存一次 checkpoint")
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

    # ─── 2. 给模型贴 LoRA 补丁 ──────────────────────────────────────────
    # target_modules: 把 LoRA 贴在 attention (q/k/v/o) + MLP (gate/up/down)
    # 全贴比只贴 attention 效果好, 显存代价小
    print(f"━━━ 加 LoRA 补丁 (r={args.lora_r}) ━━━")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,        # alpha 一般 = 2*r, 是 LoRA 缩放系数
        lora_dropout=0.0,                  # SFT 数据多时不需要 dropout
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",   # attention
            "gate_proj", "up_proj", "down_proj",       # MLP / FFN
        ],
        use_gradient_checkpointing="unsloth",   # 省显存关键, Unsloth 自家优化版
    )

    # ─── 3. 加载数据 + 套上 chat template ────────────────────────────────
    print(f"━━━ 加载数据: {args.data} ━━━")
    dataset = load_dataset("json", data_files=str(args.data), split="train")

    def format_with_template(ex):
        # 把 messages JSON → 模型期望的字符串 (含 <|im_start|>...<|im_end|> 标记)
        ex["text"] = tokenizer.apply_chat_template(
            ex["messages"],
            tokenize=False,
            add_generation_prompt=False,   # SFT 不要末尾的 prompt 引导
        )
        return ex

    dataset = dataset.map(format_with_template)
    print(f"  数据集大小: {len(dataset)}")
    print(f"  示例 (前 300 字符):\n{dataset[0]['text'][:300]}...\n")

    # ─── 调试 + 防御: 确保 tokenizer 有正确的 eos_token ─────────────────
    # Unsloth 的 4-bit Qwen 包装有时 eos_token 为 None, 直接传给 SFTConfig
    # 会被替换成占位符 <EOS_TOKEN> 然后报错. Qwen2.5 的 chat-EOS 永远是 <|im_end|>.
    QWEN_EOS = "<|im_end|>"
    print(f"  tokenizer.eos_token (loaded) = {tokenizer.eos_token!r}")
    if tokenizer.eos_token is None or tokenizer.eos_token == "<EOS_TOKEN>":
        tokenizer.eos_token = QWEN_EOS
        print(f"  → 修复: tokenizer.eos_token 设为 {QWEN_EOS!r}")
    explicit_eos = tokenizer.eos_token   # 一定是非 None 字符串了

    # ─── 4. 训练配置 ─────────────────────────────────────────────────────
    eff_batch = args.batch_size * args.grad_accum
    total_steps = (len(dataset) * args.epochs) // eff_batch
    print(f"━━━ 训练配置 ━━━")
    print(f"  per_device_batch:  {args.batch_size}")
    print(f"  grad_accum:        {args.grad_accum}")
    print(f"  effective batch:   {eff_batch}")
    print(f"  total updates ≈    {total_steps}")
    print(f"  epochs:            {args.epochs}")
    print(f"  lr:                {args.lr}")

    sft_config = SFTConfig(
        output_dir=str(args.out),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=0.03,                 # 前 3% steps 学习率从 0 慢慢升上来, 训练更稳
        lr_scheduler_type="cosine",        # 学习率随训练步衰减 (余弦)
        bf16=True,                         # L20 支持, 比 fp16 训练更稳
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        save_total_limit=2,                # 只保留最近 2 个 checkpoint, 省磁盘
        dataset_text_field="text",         # 用 dataset 里的 'text' 列做训练
        report_to="none",                  # 不上传 wandb (要的话改 "wandb")
        seed=42,
        eos_token=explicit_eos,            # 显式传 Qwen 的真实 EOS (<|im_end|>),
                                           # 新版 TRL (>=0.16) 默认值是占位符会报错
        # 注: max_seq_length 已在 FastLanguageModel.from_pretrained 时设置,
        # 新版 TRL (>=0.14) 的 SFTConfig 里移除了这个参数, 别重复设
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=sft_config,
        processing_class=tokenizer,    # 新版 TRL: tokenizer 参数改名 processing_class
    )

    # ─── 5. 关键: loss 只算 assistant 部分 ───────────────────────────────
    # Unsloth 的工具, 自动在 user 部分把 label 设为 -100 (pytorch loss 忽略)
    # Qwen2.5 ChatML 格式: <|im_start|>user\n...\n<|im_end|>\n<|im_start|>assistant\n...
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    # ─── 6. 训练 ─────────────────────────────────────────────────────────
    print(f"\n━━━ 开始训练 (预计 2-4 小时) ━━━\n")
    trainer.train()

    # ─── 7. 保存 LoRA + tokenizer ───────────────────────────────────────
    print(f"\n━━━ 保存 LoRA → {args.out} ━━━")
    model.save_pretrained(str(args.out))
    tokenizer.save_pretrained(str(args.out))

    print(f"\n✅ 训练完成")
    print(f"   LoRA 文件大小: " +
          f"{sum(p.stat().st_size for p in args.out.glob('*.safetensors'))/1e6:.1f} MB")
    print(f"   下一步: python -m src.eval --num_problems 100 --lora {args.out}")


if __name__ == "__main__":
    main()
