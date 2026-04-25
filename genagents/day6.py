"""
Generative Agents - Day 6
Multi-day simulation + paper-faithful evaluation + markdown report.

What's new vs Day 5:
  1. simulate_days(n): multi-day loop. Plans regenerate each morning via
     three-factor retrieval over accumulated memory (recent end-of-day
     reflections get pinned so yesterday's insights bleed into today).
  2. retrieve() and related callers now take an explicit `now` so the
     recency factor uses simulation time instead of wall-clock time.
     Without this, in a 3-day run recency silently decays to ~0 for all
     memories (wall-clock >> sim-time gap).
  3. evaluate_propagation(): LLM judge decides whether each agent knows
     about the seed event. Paper §5.2.2 style yes/no, plus an extended
     four-field level (L0-L3) scoring organizer/date/venue/type.
  4. compute_relationship_matrix(): for each (agent, other) pair,
     attention = sum of importance over reflections mentioning `other`,
     plus the top-3 reflection texts verbatim so readers see relationship
     quality (not a sentiment number).
  5. write_report(): emits reports/day6_report.md with the two evaluations
     side-by-side — ready to paste into a resume or project README.

Environment overrides for quick local test:
  NUM_DAYS     (default 3)
  HOURS_START  (default 7)
  HOURS_END    (default 22)

Run:
  export MINIMAX_API_KEY="..."
  NUM_DAYS=1 python day6.py     # fast check (~10 min)
  python day6.py                # full 3-day run (~30-40 min)
"""

import asyncio
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from openai import AsyncOpenAI, APIError
from sentence_transformers import SentenceTransformer


MODEL = os.environ.get("MINIMAX_MODEL", "MiniMax-M2")
BASE_URL = "https://api.minimaxi.com/v1"

REFLECTION_THRESHOLD = 30.0
N_REFLECTIONS_PER_CYCLE = 3
REFLECTION_IMPORTANCE = 8.0

# Multi-day simulation config (env-overridable)
NUM_DAYS = int(os.environ.get("NUM_DAYS", "3"))
HOURS_START = int(os.environ.get("HOURS_START", "7"))
HOURS_END = int(os.environ.get("HOURS_END", "22"))

# Reproducibility — seeds the random pair-pick in conversation triggering and any
# numpy randomness. LLM output itself is still nondeterministic.
SEED = int(os.environ.get("SEED", "42"))
random.seed(SEED)
np.random.seed(SEED)

# Token / cost tracking (populated inside _llm_call; reported at end of main).
COST = {"calls": 0, "input_tokens": 0, "output_tokens": 0}
# Approximate MiniMax-M2 pricing (¥ per 1k tokens). Verify against current
# rate card before reporting numbers in the resume; this is a sanity estimate.
INPUT_PRICE_PER_1K = 0.0012
OUTPUT_PRICE_PER_1K = 0.0080

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
            # Count tokens for cost report; usage is best-effort (server may omit)
            COST["calls"] += 1
            usage = getattr(resp, "usage", None)
            if usage is not None:
                COST["input_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
                COST["output_tokens"] += getattr(usage, "completion_tokens", 0) or 0
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
        ts = timestamp or datetime.now()
        self.memories.append(MemoryItem(
            timestamp=ts,
            type=mem_type,
            content=content,
            importance=importance,
            embedding=embed(content),
        ))
        if mem_type == "observation":
            self.importance_buffer += importance
            if self.importance_buffer >= REFLECTION_THRESHOLD:
                # Propagate the simulation time into the auto-triggered reflect
                await self.reflect(now=ts)
                self.importance_buffer = 0.0

    async def reflect(self, context_note: str = "", now: datetime | None = None):
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
                timestamp=now,
            )

    def retrieve(self, query: str, k: int = 5,
                 now: datetime | None = None) -> list[MemoryItem]:
        """Three-factor scoring. `now` defaults to wall-clock; in multi-day
        simulation always pass the simulation's current time so recency
        decays correctly relative to memory timestamps."""
        if not self.memories:
            return []
        q_emb = embed(query)
        t_now = now or datetime.now()

        def score(m: MemoryItem) -> float:
            hours_ago = (t_now - m.timestamp).total_seconds() / 3600 + 1e-3
            recency = 0.99 ** hours_ago
            relevance = cosine(m.embedding, q_emb)
            importance = m.importance / 10.0
            return recency + relevance + importance

        return sorted(self.memories, key=score, reverse=True)[:k]

    def retrieve_as_prompt(self, query: str, k: int = 5,
                           now: datetime | None = None) -> str:
        items = self.retrieve(query, k, now=now)
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

async def generate_plan(agent: Agent, date_str: str,
                        now: datetime | None = None) -> list[PlanItem]:
    """Use three-factor retrieval to surface the top-10 relevant memories
    for tomorrow's planning, plus pin the 3 most recent reflections so
    end-of-yesterday insights always make it into the prompt."""
    plan_query = f"为 {date_str} 这一天做计划,我需要考虑什么"
    retrieved = agent.retrieve(plan_query, k=10, now=now)

    # Pin the 3 most recent reflections (by timestamp, not importance).
    recent_refs = sorted(
        [m for m in agent.memories if m.type == "reflection"],
        key=lambda m: m.timestamp,
        reverse=True,
    )[:3]

    # Merge & dedupe, preserving order: retrieved first, then any missing pins.
    seen: set[int] = set()
    context_mems: list[MemoryItem] = []
    for m in retrieved:
        if id(m) not in seen:
            context_mems.append(m)
            seen.add(id(m))
    for m in recent_refs:
        if id(m) not in seen:
            context_mems.append(m)
            seen.add(id(m))

    if context_mems:
        lines = []
        for m in context_mems:
            tag = "💭" if m.type == "reflection" else "·"
            lines.append(f"{tag} [{m.importance:.0f}] {m.content}")
        context_text = "\n".join(lines)
    else:
        context_text = "(暂无)"

    prompt = PLAN_PROMPT.format(
        agent_name=agent.name,
        identity=agent.identity,
        reflections=context_text,   # legacy field name; now contains mixed memories
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
    retrieved = speaker.retrieve_as_prompt(query, k=6, now=now)

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

    # end-of-day reflection — anchor timestamp to simulation end-of-day
    print(f"\n━━━ 一天结束,反思时刻 ━━━")
    day_end_time = date.replace(hour=end_h, minute=59)
    for ag in agents:
        print(f"\n{ag.name}:")
        await ag.reflect(
            context_note=f"今天({date.date()})结束了。回顾这一天你经历的事情。",
            now=day_end_time,
        )
        # Reset importance buffer so tomorrow's buffer starts fresh.
        ag.importance_buffer = 0.0


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


# ============ multi-day loop ============

async def simulate_days(agents: list[Agent], start_date: datetime,
                        num_days: int, hours: tuple[int, int]):
    """Run `num_days` consecutive days. Regenerate each agent's plan every
    morning using three-factor retrieval over accumulated memory, so
    yesterday's end-of-day reflection influences today's plan."""
    for day_idx in range(num_days):
        current_date = start_date + timedelta(days=day_idx)
        # Anchor plan-generation retrieval to the start of this day.
        morning = current_date.replace(hour=hours[0], minute=0)
        print(f"\n\n══════════ Day {day_idx + 1} / {num_days}: {current_date.date()} ══════════")

        print(f"\n→ 生成 Day {day_idx + 1} 的日程...")
        for ag in agents:
            ag.plan = await generate_plan(ag, str(current_date.date()), now=morning)
            if len(ag.plan) < 10:
                print(f"  ⚠ {ag.name} 的 plan 只解析出 {len(ag.plan)} 条")
            print_plan(ag)

        await simulate_day(agents, current_date, hours=hours)


# ============ evaluation ============

@dataclass
class SeedEvent:
    description: str
    organizer: str
    date: str
    venue: str
    event_type: str


async def evaluate_propagation(agent: Agent, seed: SeedEvent,
                               now: datetime | None = None) -> dict:
    """Paper §5.2.2 style yes/no judgement, extended with a four-field
    level (L0-L3). LLM decides based on retrieved memories."""
    query = f"{seed.organizer} 的 {seed.event_type}"
    retrieved = agent.retrieve(query, k=10, now=now)
    mem_text = "\n".join(f"- {m.content}" for m in retrieved) or "(无相关记忆)"

    prompt = f"""判断 {agent.name} 对以下事件的了解程度。

事件: {seed.description}
关键字段:
- 组织者 (organizer): {seed.organizer}
- 日期 (date): {seed.date}
- 地点 (venue): {seed.venue}
- 类型 (type): {seed.event_type}

{agent.name} 的相关记忆(按检索分数排序):
{mem_text}

请严格输出 JSON (不要 markdown 代码块,不要任何额外文本):
{{"yes_no": "yes" 或 "no", "level": 0-3 的整数, "known_fields": [已知字段列表], "reason": "一句话说明"}}

判定标准:
- L0: 完全不知道,所有字段都不了解
- L1: 只知道"有什么活动"但不知道类型或其他细节
- L2: 知道是 party / 活动 + 至少 1 个其他字段(organizer/date/venue 之一)
- L3: 至少 3 个字段都知道

yes_no = "yes" 当且仅当 level >= 1(paper 原版:只要知道"有这件事"就算)
known_fields 从这 4 个里选: organizer, date, venue, type"""

    try:
        raw = await _llm_call(prompt, temperature=0.0)
    except APIError:
        return {"yes_no": "no", "level": 0, "known_fields": [],
                "reason": "API error during eval"}

    # Strip possible markdown fences
    raw = re.sub(r"```\w*\n?", "", raw).strip("` \n")
    # Try to find the first {...} block if there's prose around it
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        raw = m.group(0)

    try:
        result = json.loads(raw)
        # sanity defaults
        result.setdefault("yes_no", "no")
        result.setdefault("level", 0)
        result.setdefault("known_fields", [])
        result.setdefault("reason", "")
        return result
    except Exception as e:
        print(f"  ⚠ {agent.name} propagation eval JSON parse failed: {e}")
        print(f"    raw: {raw[:200]}")
        return {"yes_no": "no", "level": 0, "known_fields": [],
                "reason": f"parse failed: {raw[:80]}"}


def compute_relationship_matrix(agents: list[Agent]) -> dict:
    """For every (agent, other) pair, aggregate reflections by `agent` that
    mention `other`'s name. Returns a dict of
    {(agent_name, other_name): {count, attention, top_reflections}}."""
    matrix = {}
    for ag in agents:
        for other in agents:
            if ag is other:
                continue
            relevant = [
                m for m in ag.memories
                if m.type == "reflection" and other.name in m.content
            ]
            if not relevant:
                matrix[(ag.name, other.name)] = {
                    "count": 0,
                    "attention": 0.0,
                    "avg_importance": 0.0,
                    "top_reflections": [],
                }
                continue

            total_importance = sum(m.importance for m in relevant)
            avg_importance = total_importance / len(relevant)
            top = sorted(relevant, key=lambda m: -m.importance)[:3]
            # strip leading "[反思] " tag for cleaner display
            top_texts = [re.sub(r"^\[反思\]\s*", "", m.content) for m in top]

            matrix[(ag.name, other.name)] = {
                "count": len(relevant),
                "attention": total_importance,
                "avg_importance": avg_importance,
                "top_reflections": top_texts,
            }
    return matrix


def write_report(agents: list[Agent], seed: SeedEvent,
                 prop_results: dict, rel_matrix: dict,
                 start_date: datetime, num_days: int,
                 path: Path) -> None:
    lines: list[str] = []
    lines.append("# Day 6 Simulation Report")
    lines.append("")
    lines.append(f"- **模拟窗口**: {start_date.date()} 起 {num_days} 天")
    lines.append(f"- **Agents**: {', '.join(a.name for a in agents)}")
    lines.append(f"- **生成于**: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- **Seed 事件**: {seed.description}")
    lines.append("")

    # --- Propagation ---
    lines.append("## 1. Seed 事件传播评估")
    lines.append("")
    lines.append("### 1.1 Paper-faithful (yes/no) 主指标")
    lines.append("")
    lines.append("| Agent | 是否知道 | Level | 已知字段 | 理由 |")
    lines.append("|---|---|---|---|---|")
    for name, r in prop_results.items():
        fields = ", ".join(r.get("known_fields", [])) or "(无)"
        reason = (r.get("reason", "") or "")[:80].replace("|", "\\|")
        lines.append(f"| {name} | **{r['yes_no']}** | L{r['level']} | {fields} | {reason} |")

    yes_count = sum(1 for r in prop_results.values() if r["yes_no"] == "yes")
    l3_count = sum(1 for r in prop_results.values() if r["level"] >= 3)
    n = len(agents)
    lines.append("")
    lines.append(f"**最终覆盖率**:")
    lines.append(f"- Paper-faithful (yes/no): **{yes_count}/{n} = {yes_count/n*100:.0f}%**")
    lines.append(f"- L3 完整信息 (知道 ≥3 个字段): **{l3_count}/{n} = {l3_count/n*100:.0f}%**")
    lines.append("")

    # --- Relationship matrix ---
    lines.append("## 2. 关系矩阵")
    lines.append("")
    lines.append("每个 agent 对其他 agent 的**关注度**(总 importance)与 top-3 reflection 原文。")
    lines.append("")
    for (a_name, b_name), stats in rel_matrix.items():
        if stats["count"] == 0:
            lines.append(f"### {a_name} → {b_name}  _(无相关反思)_")
            lines.append("")
            continue
        lines.append(f"### {a_name} → {b_name}")
        lines.append(f"- **反思次数**: {stats['count']}")
        lines.append(f"- **关注度** (Σ importance): {stats['attention']:.1f}")
        lines.append(f"- **平均重要性**: {stats['avg_importance']:.2f}")
        lines.append(f"- **Top 3 reflection 原文**:")
        for i, r in enumerate(stats["top_reflections"], 1):
            lines.append(f"  {i}. {r}")
        lines.append("")

    # --- Memory stats ---
    lines.append("## 3. Memory stream 统计")
    lines.append("")
    lines.append("| Agent | 总 memory | observation | reflection |")
    lines.append("|---|---|---|---|")
    for ag in agents:
        obs = sum(1 for m in ag.memories if m.type == "observation")
        refs = sum(1 for m in ag.memories if m.type == "reflection")
        lines.append(f"| {ag.name} | {len(ag.memories)} | {obs} | {refs} |")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# ============ main ============

async def main():
    if not os.environ.get("MINIMAX_API_KEY"):
        raise SystemExit("请先 export MINIMAX_API_KEY=...")

    print(f"Config: NUM_DAYS={NUM_DAYS}, HOURS={HOURS_START}:00-{HOURS_END}:00, MODEL={MODEL}")

    agents = await build_agents()
    start_date = datetime(2026, 2, 10, 0, 0, 0)

    await simulate_days(
        agents, start_date,
        num_days=NUM_DAYS,
        hours=(HOURS_START, HOURS_END),
    )

    # ---- Evaluation ----
    print(f"\n\n══════════ 评估 ══════════")
    seed = SeedEvent(
        description="2026-02-14 Isabella 在 Hobbs Cafe 办情人节 party",
        organizer="Isabella",
        date="2026-02-14",
        venue="Hobbs Cafe",
        event_type="party",
    )

    # Propagation — anchor eval timestamp at end of the final day
    final_now = (start_date + timedelta(days=NUM_DAYS - 1)).replace(
        hour=HOURS_END, minute=59
    )
    print(f"\n→ 传播评估 (seed = {seed.description})...")
    prop_results: dict = {}
    for ag in agents:
        r = await evaluate_propagation(ag, seed, now=final_now)
        prop_results[ag.name] = r
        print(f"  {ag.name}: yes_no={r['yes_no']} L{r['level']} "
              f"fields={r.get('known_fields', [])} — {r.get('reason', '')[:60]}")

    # Relationship
    print(f"\n→ 关系矩阵...")
    rel_matrix = compute_relationship_matrix(agents)
    for (a, b), s in rel_matrix.items():
        if s["count"] == 0:
            continue
        print(f"  {a} → {b}: count={s['count']} attention={s['attention']:.1f}")

    # Report
    report_path = Path("reports/day6_report.md")
    write_report(agents, seed, prop_results, rel_matrix,
                 start_date, NUM_DAYS, report_path)
    print(f"\n📄 报告: {report_path}")

    # Brief console coverage summary
    yes_count = sum(1 for r in prop_results.values() if r["yes_no"] == "yes")
    l3_count = sum(1 for r in prop_results.values() if r["level"] >= 3)
    print(f"\n覆盖率 (paper-faithful): {yes_count}/{len(agents)} = {yes_count/len(agents)*100:.0f}%")
    print(f"覆盖率 (L3 完整):        {l3_count}/{len(agents)} = {l3_count/len(agents)*100:.0f}%")

    # Cost / reproducibility report
    cost = (COST["input_tokens"] * INPUT_PRICE_PER_1K
            + COST["output_tokens"] * OUTPUT_PRICE_PER_1K) / 1000
    print(f"\n📊 SEED={SEED}, MODEL={MODEL}")
    print(f"   API: {COST['calls']} calls, "
          f"{COST['input_tokens']:,} input + {COST['output_tokens']:,} output tokens")
    print(f"💰 估算成本: ≈ ¥{cost:.2f} "
          f"(MiniMax-M2 价格 ¥{INPUT_PRICE_PER_1K}/1k input + ¥{OUTPUT_PRICE_PER_1K}/1k output)")


if __name__ == "__main__":
    asyncio.run(main())
