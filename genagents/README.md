# Generative Agents — 复现 Stanford 2023

参考论文: [Generative Agents: Interactive Simulacra of Human Behavior (Park et al., 2023)](https://arxiv.org/abs/2304.03442)

## 进度

- [x] Day 1 — 两个 agent 对话 + 极简 memory stream (`day1.py`)
- [x] Day 2 — 三因子检索 (recency × relevance × importance) + LLM 打 importance (`day2.py`)
- [x] Day 3 — reflection 触发与生成 (`day3.py`)
- [x] Day 4 — 日计划生成 + hourly 时间推进 + 同地点触发对话 (`day4.py`)
- [x] Day 5 — 加第 3 个 agent Klaus + 信息传播实验 (`day5.py`)
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
python day3.py        # 加入 reflection,触发时控制台打印 "💭 <name> 反思: ..."
```

## 各 Day 预期差异

- **Day 2 vs Day 1**: 对话不再从第 5 轮起机械重复;importance 从全 5.0 分层到 3-9
- **Day 3 vs Day 2**: 死循环被打破 —— 当 Isabella 被连续拒绝后触发 reflection,生成 "Maria 不喜欢热闹社交" 这样的抽象判断,下一轮她的行为会转变(换话题 / 尊重选择 / 不再追问)
