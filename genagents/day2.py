"""
Generative Agents - Day 2
Three-factor memory retrieval + LLM-scored importance.

Changes from Day 1:
  - remember() is async; calls LLM to score importance (1-10) unless caller supplies one
  - each MemoryItem carries a local embedding (BGE-small-zh)
  - retrieve(query, k) replaces recent_window(n); scoring follows Park 2023 §4.3:
        score = recency + relevance + importance_normalized
  - agent_speak uses retrieval with "scene + listener" as the query

Run:
  pip install -r requirements.txt          # first time: ~1-2 GB (torch + models)
  export MINIMAX_API_KEY="..."
  python day2.py
"""

import asyncio
import os
import re
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from openai import AsyncOpenAI, APIError
from sentence_transformers import SentenceTransformer


MODEL = "MiniMax-M2"
BASE_URL = "https://api.minimaxi.com/v1"

client = AsyncOpenAI(
    api_key=os.environ.get("MINIMAX_API_KEY", ""),
    base_url=BASE_URL,
    max_retries=3,
    timeout=60.0,
)

# Local embedding model; first run downloads ~100 MB to ~/.cache/
print("→ loading embedding model (first run downloads ~100 MB)...")
EMBED_MODEL = SentenceTransformer("BAAI/bge-small-zh-v1.5")
print("✓ embedding model ready\n")

THINK_PATTERN = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def strip_thinking(text: str) -> str:
    return THINK_PATTERN.sub("", text).strip()


def embed(text: str) -> np.ndarray:
    """Normalized embedding; cosine reduces to dot product."""
    return EMBED_MODEL.encode(text, normalize_embeddings=True)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


# ============ LLM call with retry ============

async def _llm_call(prompt: str, temperature: float = 0.8) -> str:
    for attempt in range(5):
        try:
            resp = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            return strip_thinking(resp.choices[0].message.content)
        except APIError as e:
            if attempt == 4:
                raise
            wait = 2 ** attempt
            print(f"  ⚠ retry {attempt + 1}/5 after {wait}s ({type(e).__name__})")
            await asyncio.sleep(wait)


async def score_importance(content: str) -> float:
    prompt = f"""请评估以下记忆对当事人的重要性,输出 1-10 的整数。

参考锚点:
- 1-3: 日常琐事(喝咖啡、路过公园)
- 4-6: 普通社交(闲聊、打招呼、简单请求)
- 7-8: 重要事件(新认识朋友、得到承诺、收到邀请)
- 9-10: 人生大事(分手、去世、升职、重大决定)

记忆内容: "{content}"

只输出一个 1-10 的整数,不要其他任何文字。"""
    raw = await _llm_call(prompt, temperature=0.0)
    m = re.search(r"\b([1-9]|10)\b", raw)
    return float(m.group(1)) if m else 5.0


# ============ data structures ============

@dataclass
class MemoryItem:
    timestamp: datetime
    type: str
    content: str
    importance: float
    embedding: np.ndarray


@dataclass
class Agent:
    name: str
    identity: str
    memories: list[MemoryItem] = field(default_factory=list)

    async def remember(self, content: str, importance: float | None = None):
        """Append an observation.  LLM scores importance unless caller supplies one."""
        if importance is None:
            importance = await score_importance(content)
        self.memories.append(MemoryItem(
            timestamp=datetime.now(),
            type="observation",
            content=content,
            importance=importance,
            embedding=embed(content),
        ))

    def retrieve(self, query: str, k: int = 5) -> list[MemoryItem]:
        """Three-factor scoring per Park 2023 §4.3."""
        if not self.memories:
            return []
        q_emb = embed(query)
        now = datetime.now()

        def score(m: MemoryItem) -> float:
            hours_ago = (now - m.timestamp).total_seconds() / 3600 + 1e-3
            recency = 0.99 ** hours_ago
            relevance = cosine(m.embedding, q_emb)
            importance = m.importance / 10.0
            return recency + relevance + importance

        return sorted(self.memories, key=score, reverse=True)[:k]

    def retrieve_as_prompt(self, query: str, k: int = 5) -> str:
        items = self.retrieve(query, k)
        return "\n".join(f"- [{m.importance:.0f}] {m.content}" for m in items)


# ============ agent speak ============

async def agent_speak(speaker: Agent, listener: Agent, scene: str) -> str:
    query = f"{scene} 和 {listener.name} 的对话"
    retrieved = speaker.retrieve_as_prompt(query, k=5)

    prompt = f"""你是 {speaker.name}。
【人格】{speaker.identity}

【你检索到的相关记忆(按相关性+新鲜度+重要性综合排序,[]里是重要性 1-10)】
{retrieved}

【当前场景】{scene}
你正在和 {listener.name} 说话。

请用一句自然、符合人格的话回应(≤40字)。直接输出内容,不要任何前缀、引号、解释。"""

    return await _llm_call(prompt, temperature=0.8)


# ============ conversation loop ============

async def conversation(a: Agent, b: Agent, rounds: int, scene: str):
    print(f"━━━ {a.name} × {b.name} ━━━")
    print(f"场景: {scene}\n")
    speaker, listener = a, b

    for _ in range(rounds):
        utt = await agent_speak(speaker, listener, scene)
        print(f"[{speaker.name}] {utt}")

        # both sides record this turn as observations; LLM scores each
        await speaker.remember(f"我对 {listener.name} 说: {utt}")
        await listener.remember(f"{speaker.name} 对我说: {utt}")

        speaker, listener = listener, speaker

    print(f"\n━━━ 对话结束 ({rounds} 轮) ━━━")


# ============ entry ============

async def build_agents() -> tuple[Agent, Agent]:
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
    await isabella.remember(
        "我打算 2026-02-14 在 Hobbs Cafe 办情人节 party,想邀请常客",
        importance=9.0,
    )
    await isabella.remember(
        "Maria 是 Cafe 的常客,经常一个人来学习,我一直想和她多聊聊",
        importance=7.0,
    )
    return isabella, maria


async def main():
    if not os.environ.get("MINIMAX_API_KEY"):
        raise SystemExit("请先 export MINIMAX_API_KEY=...")

    isabella, maria = await build_agents()

    await conversation(
        isabella, maria,
        rounds=8,
        scene=(
            "2026年2月10日傍晚的 Hobbs Cafe,"
            "Maria 像往常一样来喝咖啡,Isabella 走过去招呼她"
        ),
    )

    for agent in (isabella, maria):
        print(f"\n═══ {agent.name} 的记忆 ({len(agent.memories)} 条,按重要性降序) ═══")
        for m in sorted(agent.memories, key=lambda x: -x.importance):
            print(f"  [{m.importance:.1f}] {m.content}")


if __name__ == "__main__":
    asyncio.run(main())
