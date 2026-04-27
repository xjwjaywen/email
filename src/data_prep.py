"""
data_prep.py — 准备 MATH SFT 训练数据

干啥的:
    从 HuggingFace 上的 MATH 数据集加载, 把每道题的 (problem, solution)
    转成 ChatML 格式, 写入 jsonl, 用于 SFT.

输入:  lighteval/MATH (从 HF 自动下载, 走你的 HF_ENDPOINT 镜像)
输出:  data/sft_train.jsonl  每行: {"messages": [...]}

跑法:
    python -m src.data_prep --split train --out data/sft_train.jsonl
    python -m src.data_prep --split test  --out data/sft_test.jsonl --max_samples 200
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset


# 评测时 eval.py 也用同一个 system prompt, 训练-评测要一致, 否则模型不知道什么时候用 \boxed{}
SYSTEM_PROMPT = (
    "You are a math problem solver. Solve the problem step by step, "
    "showing your reasoning. Put your final answer inside \\boxed{}."
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "test"],
                        help="MATH 数据集 split, train ~7500 题, test ~5000 题")
    parser.add_argument("--out", type=Path, default=Path("data/sft_train.jsonl"),
                        help="输出 jsonl 文件路径")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="只取前 N 条 (调试用), 默认全量")
    parser.add_argument("--dataset", default="lighteval/MATH",
                        help="HF 数据集名 (lighteval/MATH 是稳定镜像, "
                             "原 hendrycks/competition_math 已下架)")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"加载 {args.dataset} ({args.split})...")
    ds = load_dataset(args.dataset, split=args.split)
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    print(f"写入 {len(ds)} 条 → {args.out}")
    with open(args.out, "w", encoding="utf-8") as f:
        for ex in ds:
            # MATH 字段: problem (题目), solution (含完整解题过程 + \boxed{答案}),
            #            level (难度 1-5), type (代数/几何/数论/...)
            messages = [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": ex["problem"].strip()},
                {"role": "assistant", "content": ex["solution"].strip()},
            ]
            f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")

    print(f"✅ 完成. 文件大小: {args.out.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
