"""
Generative Agents - Day 1
Two agents have a conversation; each stores observations into a minimal memory stream.

Goal of Day 1:
  - Prove the API + prompt + memory-append loop works end-to-end.
  - Do NOT add retrieval, reflection, planning yet — those come in Day 2-3.

Run:
  export MINIMAX_API_KEY="..."
  python day1.py
"""

import asyncio
import os
from dataclasses import dataclass, field
from datetime import datetime

from openai import AsyncOpenAI


MODEL = "MiniMax-M2"
BASE_URL = "https://api.minimaxi.com/v1"   # 国内站; 国际站用 https://api.minimax.io/v1

client = AsyncOpenAI(
    api_key=os.environ.get("MINIMAX_API_KEY", ""),
    base_url=BASE_URL,
)


# ============ data structures ============

@dataclass
class MemoryItem:
    timestamp: datetime
    type: str          # "observation" | (later) "reflection" | "plan"
    content: str
    importance: float = 5.0   # 1-10; Day 1 hardcoded, Day 3 LLM-scored


@dataclass
class Agent:
    name: str
    identity: str      # persona prompt
    memories: list[MemoryItem] = field(default_factory=list)

    def remember(self, content: str, importance: float = 5.0):
        self.memories.append(MemoryItem(
            timestamp=datetime.now(),
            type="observation",
            content=content,
            importance=importance,
        ))

    def recent_window(self, n: int = 10) -> str:
        """Day 1: naive 'last N items' retrieval.
        Day 2 will replace this with three-factor scoring
        (recency * relevance * importance)."""
        recent = self.memories[-n:]
        return "\n".join(f"- {m.content}" for m in recent)


# ============ agent speak ============

async def agent_speak(speaker: Agent, listener: Agent, scene: str) -> str:
    prompt = f"""你是 {speaker.name}。
【人格】{speaker.identity}

【你最近的记忆】
{speaker.recent_window(10)}

【当前场景】{scene}
你正在和 {listener.name} 说话。

请用一句自然、符合人格的话回应(≤40字)。直接输出内容,不要任何前缀、引号、解释。"""

    resp = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
    )
    return resp.choices[0].message.content.strip()


# ============ conversation loop ============

async def conversation(a: Agent, b: Agent, rounds: int, scene: str):
    print(f"\n━━━ {a.name} × {b.name} ━━━")
    print(f"场景: {scene}\n")
    speaker, listener = a, b

    for _ in range(rounds):
        utt = await agent_speak(speaker, listener, scene)
        print(f"[{speaker.name}] {utt}")

        # both sides record this turn as an observation
        speaker.remember(f"我对 {listener.name} 说: {utt}")
        listener.remember(f"{speaker.name} 对我说: {utt}")

        speaker, listener = listener, speaker

    print(f"\n━━━ 对话结束 ({rounds} 轮) ━━━")


# ============ entry ============

def build_agents() -> tuple[Agent, Agent]:
    isabella = Agent(
        name="Isabella",
        identity=(
            "32岁,Hobbs Cafe 老板娘,热情外向。"
            "最近在策划 2026 年情人节 party,想邀请常客一起庆祝。"
        ),
    )
    maria = Agent(
        name="Maria",
        identity=(
            "25岁,计算机系大学生,性格内向但对熟人友好,"
            "喜欢下午一个人在 Hobbs Cafe 喝咖啡、写代码。"
        ),
    )

    # Seed Isabella with 'plan-ish' memories so she has something to talk about.
    # In Day 3 these will be generated via a proper planning step.
    isabella.remember(
        "我打算 2026-02-14 在 Hobbs Cafe 办情人节 party,想邀请常客",
        importance=9.0,
    )
    isabella.remember(
        "Maria 是 Cafe 的常客,经常一个人来学习,我一直想和她多聊聊",
        importance=7.0,
    )
    return isabella, maria


async def main():
    if not os.environ.get("MINIMAX_API_KEY"):
        raise SystemExit("请先 export MINIMAX_API_KEY=...")

    isabella, maria = build_agents()

    await conversation(
        isabella, maria,
        rounds=8,
        scene=(
            "2026年2月10日傍晚的 Hobbs Cafe,"
            "Maria 像往常一样来喝咖啡,Isabella 走过去招呼她"
        ),
    )

    print(f"\n═══ {isabella.name} 的记忆 ({len(isabella.memories)} 条) ═══")
    for m in isabella.memories:
        print(f"  [{m.importance:.1f}] {m.content}")
    print(f"\n═══ {maria.name} 的记忆 ({len(maria.memories)} 条) ═══")
    for m in maria.memories:
        print(f"  [{m.importance:.1f}] {m.content}")


if __name__ == "__main__":
    asyncio.run(main())
