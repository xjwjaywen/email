# Generative Agents — 复现 Stanford 2023

参考论文: [Generative Agents: Interactive Simulacra of Human Behavior (Park et al., 2023)](https://arxiv.org/abs/2304.03442)

## 进度

- [x] Day 1 — 两个 agent 对话 + 极简 memory stream (`day1.py`)
- [x] Day 2 — 三因子检索 (recency × relevance × importance) + LLM 打 importance (`day2.py`)
- [ ] Day 3 — reflection 触发与生成
- [ ] Day 4 — plan 生成 + 时间推进
- [ ] Day 5 — 加第 3 个 agent,测试信息传播
- [ ] Day 6 — world model (地点/移动/时间系统)
- [ ] Day 7 — 简单评估 (信息扩散 + 人格一致性)

## 运行

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt     # Day 2 起会装 torch+sentence-transformers,约 1-2 GB

export MINIMAX_API_KEY="你的 key"

python day1.py        # naive last-N retrieval,对照组
python day2.py        # 三因子检索,第一次跑额外下载 ~100 MB embedding 模型
```

## Day 2 vs Day 1 预期差异

- 对话不再从第 5 轮开始机械重复(相关性会把不同主题的记忆挤进来)
- memory 里 importance 不再全是 5.0,seed 的 plan(9)、party 邀请(约 7-8)、日常寒暄(约 3-5)会分层
