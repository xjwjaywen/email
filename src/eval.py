"""
eval.py — 在 MATH 测试集上评测模型

干啥的:
    加载 base 模型 (可选 + LoRA), 让它解 MATH test 题, 从输出抠出 \\boxed{X},
    和 ground truth 的 \\boxed{X} 比对算准确率. 完全程序化判分, 不要 API.

跑法 (避开别人占用的 GPU):
    CUDA_VISIBLE_DEVICES=3 python -m src.eval --num_problems 100
    CUDA_VISIBLE_DEVICES=3 python -m src.eval --num_problems 100 --lora outputs/v01_sft

baseline 期望 ~50% (Qwen2.5-7B-Instruct 公开数字).
100 题大约 30-60 分钟 (取决于解答长度).
"""

import argparse
import json
import re
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from unsloth import FastLanguageModel


SYSTEM_PROMPT = (
    "You are a math problem solver. Solve the problem step by step, "
    "showing your reasoning. Put your final answer inside \\boxed{}."
)


def extract_boxed_answer(text: str) -> str | None:
    """从一段文字里抠出最后一个 \\boxed{X} 中的 X.

    简单实现, 只处理一层嵌套花括号 (\\boxed{a^{b}} 这种). 对于更复杂的
    \\boxed{\\frac{1}{\\sqrt{2}}} 也能处理因为正则递归一层. v0.2 升级
    sympy 等价判断时这个函数也要换成更鲁棒的版本.
    """
    # 匹配 \boxed{...}, 内部允许一层嵌套花括号
    pattern = r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}"
    matches = re.findall(pattern, text)
    return matches[-1].strip() if matches else None


def is_equivalent(pred: str | None, gold: str | None) -> bool:
    """判断两个答案是否等价. v0.1 用最简字符串比对(去空格 + 大小写).

    后期 v0.2 升级:
        from sympy.parsing.latex import parse_latex
        return sympy.simplify(parse_latex(pred) - parse_latex(gold)) == 0
    这能识别 1/2 == 0.5 == 2^{-1} 这种数学等价.
    """
    if pred is None or gold is None:
        return False
    return pred.replace(" ", "").lower() == gold.replace(" ", "").lower()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora", type=Path, default=None,
                        help="LoRA 文件夹, 不给就跑 base 模型 baseline")
    parser.add_argument("--num_problems", type=int, default=100,
                        help="评测多少题, 100 题 ~30-60 分钟, 全量 5000 题数小时")
    parser.add_argument("--model", default="unsloth/Qwen2.5-7B-Instruct",
                        help="base 模型")
    parser.add_argument("--out", type=Path, default=Path("outputs/eval_results.json"),
                        help="评测结果保存路径")
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                        help="模型生成最长长度, MATH 解答可以很长")
    parser.add_argument("--dataset", default="lighteval/MATH")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # ─── 1. 加载模型 ─────────────────────────────────────────────────────
    print(f"加载 {args.model}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        args.model,
        max_seq_length=4096,           # 评测要长一点 (题 + 解答)
        load_in_4bit=True,
        device_map={"": 0},            # 强制全放第 0 张可见 GPU
    )

    if args.lora:
        print(f"加载 LoRA: {args.lora}")
        model.load_adapter(str(args.lora))

    # 切到推理模式 (Unsloth 会做一些优化)
    FastLanguageModel.for_inference(model)

    # ─── 2. 加载数据 ─────────────────────────────────────────────────────
    print(f"加载 {args.dataset} test 前 {args.num_problems} 题...")
    ds = load_dataset(args.dataset, split="test")
    ds = ds.select(range(min(args.num_problems, len(ds))))

    # ─── 3. 推理 + 判分循环 ──────────────────────────────────────────────
    results = []
    correct = 0
    for ex in tqdm(ds, desc="评测中"):
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": ex["problem"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,                # 贪心解码, 评测要可复现
                pad_token_id=tokenizer.pad_token_id,
            )
        # 只解码新生成的部分, 不要 prompt
        response = tokenizer.decode(
            out[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )

        pred = extract_boxed_answer(response)
        gold = extract_boxed_answer(ex["solution"])
        ok = is_equivalent(pred, gold)
        correct += int(ok)

        results.append({
            "problem":  ex["problem"][:300],   # 截断省空间
            "level":    ex.get("level", ""),
            "type":     ex.get("type", ""),
            "pred":     pred,
            "gold":     gold,
            "correct":  ok,
            # 可选: 把完整 response 也存下来调试用
            "response": response[:500],
        })

    # ─── 4. 统计 + 保存 ──────────────────────────────────────────────────
    accuracy = correct / len(ds)

    setup = "baseline (no LoRA)" if args.lora is None else f"with LoRA: {args.lora}"
    payload = {
        "setup":      setup,
        "model":      args.model,
        "n":          len(ds),
        "correct":    correct,
        "accuracy":   accuracy,
        "details":    results,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print()
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  Accuracy: {accuracy:.1%}  ({correct}/{len(ds)})")
    print(f"  Setup:    {setup}")
    print(f"  保存到:    {args.out}")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


if __name__ == "__main__":
    main()
