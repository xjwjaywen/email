"""聊天/问答 API 路由。"""

import json
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from app.database import get_db
from app.models.conversation import ChatResponse, MessageCreate, MessageOut
from app.services.llm import generate_answer, get_llm_client, get_chat_model
from app.services.memory import extract_memory_from_conversation, get_memory_context

router = APIRouter(prefix="/api/conversations", tags=["chat"])


async def auto_title(conv_id: str, user_message: str):
    """首次对话时自动生成对话标题。"""
    db = await get_db()
    try:
        # 检查是否是首条消息（只有刚插入的 user 消息）
        cursor = await db.execute(
            "SELECT COUNT(*) FROM messages WHERE conversation_id = ? AND role = 'user'",
            (conv_id,),
        )
        count = (await cursor.fetchone())[0]
        if count != 1:
            return

        # 检查标题是否还是默认的
        cursor = await db.execute(
            "SELECT title FROM conversations WHERE id = ?", (conv_id,)
        )
        row = await cursor.fetchone()
        if not row or row[0] != "New Conversation":
            return

        # 用 LLM 生成短标题
        try:
            client = get_llm_client()
            model = get_chat_model()
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "根据用户的提问，生成一个简短的对话标题（不超过15个字，不要加引号和标点）。"},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.3,
            )
            title = (resp.choices[0].message.content or "").strip().strip('"\'""''')
            # 清理 <think> 标签
            import re
            title = re.sub(r"<think>[\s\S]*?</think>\s*", "", title).strip()
            if title and len(title) <= 50:
                now = datetime.now(timezone.utc).isoformat()
                await db.execute(
                    "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                    (title, now, conv_id),
                )
                await db.commit()
        except Exception:
            pass
    finally:
        await db.close()


@router.post("/{conv_id}/messages", response_model=ChatResponse)
async def send_message(conv_id: str, data: MessageCreate):
    db = await get_db()
    try:
        # 验证对话存在
        cursor = await db.execute(
            "SELECT id FROM conversations WHERE id = ?", (conv_id,)
        )
        if not await cursor.fetchone():
            raise HTTPException(status_code=404, detail="Conversation not found")

        now = datetime.now(timezone.utc).isoformat()

        # 保存用户消息
        user_msg_id = str(uuid.uuid4())
        await db.execute(
            "INSERT INTO messages (id, conversation_id, role, content, sources, created_at) "
            "VALUES (?, ?, 'user', ?, NULL, ?)",
            (user_msg_id, conv_id, data.content, now),
        )
        await db.commit()

        # 获取对话历史
        cursor = await db.execute(
            "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY created_at",
            (conv_id,),
        )
        rows = await cursor.fetchall()
        history = [{"role": r["role"], "content": r["content"]} for r in rows]

        # 获取 memory 上下文
        memory_ctx = await get_memory_context()

        # 生成回答
        answer, sources = generate_answer(
            query=data.content,
            conversation_history=history,
            memory_context=memory_ctx,
        )

        # 保存助手消息
        assistant_msg_id = str(uuid.uuid4())
        sources_json = json.dumps(sources, ensure_ascii=False)
        now2 = datetime.now(timezone.utc).isoformat()
        await db.execute(
            "INSERT INTO messages (id, conversation_id, role, content, sources, created_at) "
            "VALUES (?, ?, 'assistant', ?, ?, ?)",
            (assistant_msg_id, conv_id, answer, sources_json, now2),
        )

        # 更新对话时间
        await db.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?", (now2, conv_id)
        )
        await db.commit()

        # 自动生成标题 + 提取 memory（不阻塞响应）
        try:
            await auto_title(conv_id, data.content)
        except Exception:
            pass
        try:
            await extract_memory_from_conversation(conv_id, data.content, answer)
        except Exception:
            pass

        user_msg = MessageOut(
            id=user_msg_id,
            conversation_id=conv_id,
            role="user",
            content=data.content,
            sources=[],
            created_at=now,
        )
        assistant_msg = MessageOut(
            id=assistant_msg_id,
            conversation_id=conv_id,
            role="assistant",
            content=answer,
            sources=sources,
            created_at=now2,
        )

        return ChatResponse(user_message=user_msg, assistant_message=assistant_msg)
    finally:
        await db.close()
