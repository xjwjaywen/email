"""
Generative Agents - Day 5
Third agent (Klaus) + information-propagation experiment.

What's new vs Day 4:
  - Klaus joins: Isabella's long-time friend, Cafe regular, vaguely knows Maria
  - simulate_day now handles N>=3 agents. When 2+ agents share a location,
    we randomly pick ONE pair for a conversation that hour (keeps API cost
    bounded and mirrors two-person convos in a real cafe).
  - Isabella and Klaus have mutual seed memories ("old friend").
  - Klaus has a weak seed memory about Maria ("cafe regular, never talked").

Experiment to watch:
  1. Isabella tells Klaus about the party (usually early, at cafe).
  2. Later, Klaus runs into Maria somewhere.
  3. WILL KLAUS SPONTANEOUSLY RELAY THE PARTY NEWS TO MARIA?
     - If yes: information propagation has emerged (paper §5)
     - If no: retrieval during Klaus↔Maria conversation didn't surface
       the Isabella memory; adjust weights or prompt.

Run:
  export MINIMAX_API_KEY="..."
  python day5.py
"""

import asyncio
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
from openai import AsyncOpenAI, APIError
from sentence_transformers import SentenceTransformer


MODEL = os.environ.get("MINIMAX_MODEL", "MiniMax-M2")
BASE_URL = "https://api.minimaxi.com/v1"

REFLECTION_THRESHOLD = 30.0
N_REFLECTIONS_PER_CYCLE = 3
REFLECTION_IMPORTANCE = 8.0

# Conversation trigger
CONVERSATION_PROBABILITY = 0.8   # when co-located, chance to chat this hour
CONVERSATION_ROUNDS = 4          # rounds per triggered chat (shorter than Day 3)

client = AsyncOpenAI(
    api_key=os.environ.get("MINIMAX_API_KEY", ""),
    base_url=BASE_URL,
    max_retries=3,
    timeout=60.0,
)

print("→ loading embedding model...")
EMBED_MODEL = SentenceTransformer("BAAI/bge-small-zh-v1.5")
print("✓ embedding model ready\n")

THINK_PATTERN = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def strip_thinking(text: str) -> str:
    return THINK_PATTERN.sub("", text).strip()


def embed(text: str) -> np.ndarray:
    return EMBED_MODEL.encode(text, normalize_embeddings=True)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


# ============ LLM ============

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

- 1-3: 日常琐事(喝咖啡、路过公园)
- 4-6: 普通社交(闲聊、打招呼、简单请求)
- 7-8: 重要事件(新认识朋友、得到承诺、收到邀请)
- 9-10: 人生大事(分手、去世、升职、重大决定)

记忆内容: "{content}"

只输出一个 1-10 的整数,不要其他任何文字。"""
    try:
        raw = await _llm_call(prompt, temperature=0.0)
    except APIError:
        print("  ⚠ importance scoring failed, defaulting to 5.0")
        return 5.0
    m = re.search(r"\b([1-9]|10)\b", raw)
    return float(m.group(1)) if m else 5.0


REFLECTION_PROMPT = """基于 {agent_name} 最近经历的重要观察,请总结 {n} 条关于对方 / 自己 / 当前情况的高层判断或模式。

{agent_name} 已经有过的反思判断(**不要重复或同义改写这些**,要带来新角度):
{existing_reflections}

{agent_name} 最近的重要观察(按重要性降序):
{memories}

要求:
1. 每条必须是抽象的陈述句,不是对某次事件的简单复述
2. **不能是已有反思的同义改写** — 每条要带来新信息、新视角或新推断
3. 应该能指导 {agent_name} 在未来的决策中如何行动
4. 可以是对某人性格的判断、某类情况的规律、自己的情绪状态、关系变化等

好例子: "Maria 对热闹社交回避";"我追问 party 让 Maria 有压力";"Maria 虽然嘴上说'有空再来',其实对邀请心动"
不好例子: "Maria 说过'我考虑一下'"(太具体);
          "Maria 喜欢安静"(如已有 "Maria 偏好独处" 就是同义改写);
          "Klaus 上周点了拿铁"(琐事)

输出格式 — 每行形如 "N. 你的判断内容"(禁止使用任何 <> 括号或 XML 标签):
示例(照这种形式,不要照搬内容):
1. Maria 对热闹社交回避
2. 我追问 party 让 Maria 有压力
3. Maria 虽然嘴上说"有空再来",其实对邀请心动

现在请输出 {n} 条 {agent_name} 自己的新判断,每条独立一行,带编号,**不要 <> 标签,不要前缀,不要解释**。"""


PLAN_PROMPT = """你是 {agent_name}。
【你的人格】{identity}

【你已有的相关 reflection(如果有)】
{reflections}

今天是 {date}。请为你自己制定一份 24 小时的日程安排,以**小时**为粒度。

输出必须是 24 行,每行严格格式为:
HH:00 | 地点 | 活动

可用地点(只能从这几个里选,**地点字符串必须完全照抄**,连"的家"前缀都不能漏):
- {agent_name}的家(你自己的公寓/住处,只有你能去)
- Hobbs Cafe(公共咖啡馆)
- 图书馆(公共)
- 公园(公共)
- 街上(通勤/散步)

要求:
- 作息要符合你的人格(内向者可能大部分时间在 {agent_name}的家/图书馆;外向经营者大部分时间在 Cafe)
- 地点切换要自然(不要每小时都跳来跳去)
- 0:00-6:00 一般在 {agent_name}的家 睡觉
- 不要解释,只输出 24 行

示例格式(注意"的家"前缀):
00:00 | {agent_name}的家 | 睡觉
01:00 | {agent_name}的家 | 睡觉
...
08:00 | 街上 | 去咖啡馆通勤
09:00 | Hobbs Cafe | 准备开店
..."""


# ============ data structures ============

@dataclass
class MemoryItem:
    timestamp: datetime
    type: str
    content: str
    importance: float
    embedding: np.ndarray


@dataclass
class PlanItem:
    hour: int               # 0-23
    location: str
    activity: str


@dataclass
class Agent:
    name: str
    identity: str
    memories: list[MemoryItem] = field(default_factory=list)
    importance_buffer: float = 0.0
    plan: list[PlanItem] = field(default_factory=list)   # length == 24

    async def remember(self, content: str, importance: float | None = None,
                       mem_type: str = "observation", timestamp: datetime | None = None):
        if importance is None:
            importance = await score_importance(content)
        self.memories.append(MemoryItem(
            timestamp=timestamp or datetime.now(),
            type=mem_type,
            content=content,
            importance=importance,
            embedding=embed(content),
        ))
        if mem_type == "observation":
            self.importance_buffer += importance
            if self.importance_buffer >= REFLECTION_THRESHOLD:
                await self.reflect()
                self.importance_buffer = 0.0

    async def reflect(self, context_note: str = ""):
        # Only observations feed the source pool — reflecting on reflections
        # is what made Day 4 output so repetitive.
        observations = [m for m in self.memories if m.type == "observation"]
        top_obs = sorted(observations, key=lambda m: -m.importance)[:20]
        mem_text = "\n".join(f"- [{m.importance:.0f}] {m.content}" for m in top_obs) or "(暂无观察)"

        # Existing reflections are shown separately with a "don't repeat" instruction.
        existing = [m for m in self.memories if m.type == "reflection"]
        existing_text = "\n".join(f"- {m.content}" for m in existing[-10:]) or "(暂无)"

        prompt = REFLECTION_PROMPT.format(
            agent_name=self.name,
            n=N_REFLECTIONS_PER_CYCLE,
            memories=mem_text,
            existing_reflections=existing_text,
        )
        if context_note:
            prompt = context_note + "\n\n" + prompt

        raw = await _llm_call(prompt, temperature=0.3)
        reflections = re.findall(r"\d+[\.\)]\s*(.+)", raw)
        cleaned = []
        for r in reflections:
            # strip <...> tags if LLM put them in despite instructions
            r = re.sub(r"</?[^>]+>", "", r).strip()
            # drop empty / placeholder outputs
            if len(r) < 8:
                continue
            cleaned.append(r)
        reflections = cleaned[:N_REFLECTIONS_PER_CYCLE]

        if not reflections:
            print(f"  ⚠ {self.name} 反思输出为空或无效,跳过")
            return

        print(f"  💭 {self.name} 反思:")
        for r in reflections:
            print(f"     - {r}")
            await self.remember(
                content=f"[反思] {r}",
                importance=REFLECTION_IMPORTANCE,
                mem_type="reflection",
            )

    def retrieve(self, query: str, k: int = 5) -> list[MemoryItem]:
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
        lines = []
        for m in items:
            tag = "💭" if m.type == "reflection" else "·"
            lines.append(f"{tag} [{m.importance:.0f}] {m.content}")
        return "\n".join(lines)

    def current_plan_item(self, hour: int) -> PlanItem | None:
        for p in self.plan:
            if p.hour == hour:
                return p
        return None


# ============ plan generation ============

async def generate_plan(agent: Agent, date_str: str) -> list[PlanItem]:
    # Gather existing reflections (if any) to inform the plan
    refs = [m for m in agent.memories if m.type == "reflection"]
    if refs:
        ref_text = "\n".join(f"- {m.content}" for m in refs[:5])
    else:
        ref_text = "(暂无)"

    prompt = PLAN_PROMPT.format(
        agent_name=agent.name,
        identity=agent.identity,
        reflections=ref_text,
        date=date_str,
    )
    raw = await _llm_call(prompt, temperature=0.5)

    # parse "HH:00 | location | activity" lines
    plan = []
    for line in raw.split("\n"):
        line = line.strip()
        m = re.match(r"(\d{1,2}):\d{2}\s*[|｜]\s*(.+?)\s*[|｜]\s*(.+)", line)
        if not m:
            continue
        hour = int(m.group(1))
        if 0 <= hour <= 23:
            plan.append(PlanItem(hour=hour, location=m.group(2).strip(), activity=m.group(3).strip()))
    return plan


# ============ conversation ============

async def agent_speak(speaker: Agent, listener: Agent, scene: str,
                      now: datetime,
                      dialogue_so_far: list[tuple[str, str]] | None = None) -> str:
    """dialogue_so_far: list of (speaker_name, utterance) for THIS conversation
    so far — the in-turn history that prevents "starting over" each round."""
    query = f"{scene} 和 {listener.name} 的对话"
    retrieved = speaker.retrieve_as_prompt(query, k=6)

    # Build a time-anchor sentence so the LLM doesn't invent "下周" vs "下个月"
    valentines = datetime(now.year, 2, 14)
    days_to_vday = (valentines - now).days
    if days_to_vday >= 0:
        vday_hint = f"今天是 {now.date()},距情人节(02-14) 还有 {days_to_vday} 天。"
    else:
        vday_hint = f"今天是 {now.date()},情人节(02-14)已过 {-days_to_vday} 天。"

    # Render turn-by-turn history of THIS conversation
    if dialogue_so_far:
        hist_lines = [f"{name}: {utt}" for name, utt in dialogue_so_far]
        hist_block = "\n".join(hist_lines)
        dialog_note = (
            f"\n【当前这次对话刚才说过的(按顺序)】\n{hist_block}\n"
            f"(不要重新打招呼、不要重新邀请、不要当成刚相遇 — 在这个对话上下文里接下去说)"
        )
    else:
        dialog_note = "\n(这是你们本次对话的开场第一句)"

    prompt = f"""你是 {speaker.name}。
【人格】{speaker.identity}
【时间锚点】{vday_hint}

【你检索到的相关长期记忆(💭 = 反思判断, · = 具体观察)】
{retrieved}

【当前场景】{scene}
你正在和 {listener.name} 说话。{dialog_note}

请用一句自然、符合人格的话回应(≤40字)。如果提到日期/时间,必须基于上面【时间锚点】。
直接输出一句话,不要任何前缀、引号、叙事描写、XML 标签。"""
    return await _llm_call(prompt, temperature=0.8)


async def short_conversation(a: Agent, b: Agent, scene: str, rounds: int,
                             now: datetime):
    dialogue_so_far: list[tuple[str, str]] = []
    speaker, listener = a, b
    for _ in range(rounds):
        utt = await agent_speak(speaker, listener, scene, now,
                                dialogue_so_far=dialogue_so_far)
        print(f"     [{speaker.name}] {utt}")

        # record in THIS conversation's running transcript
        dialogue_so_far.append((speaker.name, utt))

        shared_score = await score_importance(f"对话内容: {utt}")
        await speaker.remember(
            f"我对 {listener.name} 说: {utt}",
            importance=shared_score,
            timestamp=now,
        )
        await listener.remember(
            f"{speaker.name} 对我说: {utt}",
            importance=shared_score,
            timestamp=now,
        )
        speaker, listener = listener, speaker


# ============ day simulation ============

async def simulate_day(agents: list[Agent], date: datetime, hours: tuple[int, int] = (7, 22)):
    """Run the day from hours[0] to hours[1] inclusive (default 7am-10pm).
    Skips sleeping hours to save API cost."""
    start_h, end_h = hours
    for hour in range(start_h, end_h + 1):
        now = date.replace(hour=hour, minute=0, second=0, microsecond=0)
        print(f"\n━━━ {hour:02d}:00 ━━━")

        # Each agent records where they are
        for ag in agents:
            item = ag.current_plan_item(hour)
            if item is None:
                continue
            line = f"{ag.name}: 在 {item.location} {item.activity}"
            print(f"  {line}")
            # agent logs this as an observation about themselves
            await ag.remember(
                f"我在 {item.location} {item.activity}",
                timestamp=now,
            )

        # Find co-located pairs
        loc_map: dict[str, list[Agent]] = {}
        for ag in agents:
            item = ag.current_plan_item(hour)
            if item is None:
                continue
            loc_map.setdefault(item.location, []).append(ag)

        for loc, occupants in loc_map.items():
            if len(occupants) >= 2 and random.random() < CONVERSATION_PROBABILITY:
                # With 3+ people co-located (e.g. Isabella + Maria + Klaus all
                # at Hobbs Cafe), pick a random pair. Real cafes don't have
                # everyone in one conversation; pairwise is closer to reality
                # and keeps API cost bounded.
                a, b = random.sample(occupants, 2)
                scene = f"{hour:02d}:00 在 {loc},{a.name} 和 {b.name} 相遇"
                other_names = [x.name for x in occupants if x not in (a, b)]
                if other_names:
                    scene += f"(此时 {'、'.join(other_names)} 也在场但没加入对话)"
                print(f"  💬 对话触发 @ {loc}: {a.name} × {b.name}")
                await short_conversation(a, b, scene, CONVERSATION_ROUNDS, now)

    # end-of-day reflection
    print(f"\n━━━ 一天结束,反思时刻 ━━━")
    for ag in agents:
        print(f"\n{ag.name}:")
        await ag.reflect(context_note=f"今天({date.date()})结束了。回顾这一天你经历的事情。")


# ============ entry ============

async def build_agents() -> list[Agent]:
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
    klaus = Agent(
        name="Klaus",
        identity=(
            "38岁,自由摄影师,Hobbs Cafe 的老常客,性格温和健谈。"
            "和 Isabella 是十多年的朋友,经常去 Cafe 喝咖啡拍街拍。"
            "认识 Maria 但不熟,只在 Cafe 见过几次。"
        ),
    )

    # Isabella's seeds
    await isabella.remember(
        "我打算 2026-02-14 在 Hobbs Cafe 办情人节 party,想邀请常客",
        importance=9.0,
    )
    await isabella.remember(
        "Maria 是 Cafe 的常客,经常一个人来学习,我一直想和她多聊聊",
        importance=7.0,
    )
    await isabella.remember(
        "Klaus 是我十多年的老朋友,经常来 Cafe,是我最信任的常客之一",
        importance=8.0,
    )

    # Klaus's seeds — he knows Isabella well, knows Maria vaguely.
    # Critically: Klaus has NO seed memory about the party itself.
    # Whether he learns about it must happen through the simulated conversations.
    await klaus.remember(
        "Isabella 是 Hobbs Cafe 的老板娘,我们是十多年的老朋友",
        importance=8.0,
    )
    await klaus.remember(
        "Maria 是 Cafe 的另一个常客,我见过她几次,但没深聊过",
        importance=5.0,
    )

    return [isabella, maria, klaus]


def print_plan(ag: Agent):
    print(f"\n=== {ag.name} 的日程 ===")
    for p in ag.plan:
        print(f"  {p.hour:02d}:00 | {p.location} | {p.activity}")


async def main():
    if not os.environ.get("MINIMAX_API_KEY"):
        raise SystemExit("请先 export MINIMAX_API_KEY=...")

    agents = await build_agents()
    today = datetime(2026, 2, 10, 0, 0, 0)

    # generate plans
    for ag in agents:
        print(f"→ 生成 {ag.name} 的日程...")
        ag.plan = await generate_plan(ag, str(today.date()))
        print_plan(ag)
        if len(ag.plan) < 10:
            print(f"  ⚠ {ag.name} 的 plan 只解析出 {len(ag.plan)} 条,可能 prompt 格式有问题")

    # run the day (7am - 10pm to save API calls)
    await simulate_day(agents, today, hours=(7, 22))

    # dump final memory
    for ag in agents:
        obs = [m for m in ag.memories if m.type == "observation"]
        refs = [m for m in ag.memories if m.type == "reflection"]
        print(f"\n═══ {ag.name} 全天 memory ({len(ag.memories)} 条: {len(obs)} obs + {len(refs)} ref) ═══")
        for m in sorted(ag.memories, key=lambda x: -x.importance):
            tag = "💭" if m.type == "reflection" else "·"
            print(f"  {tag} [{m.importance:.1f}] {m.content}")


if __name__ == "__main__":
    asyncio.run(main())
