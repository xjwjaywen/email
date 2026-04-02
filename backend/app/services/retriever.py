"""检索服务：语义检索 + 关键词匹配混合检索。"""

import re

from app.config import RETRIEVAL_TOP_K
from app.services.indexer import get_collection


def _extract_keywords(query: str) -> list[str]:
    """从查询中提取有意义的关键词。支持中文多粒度切分。"""
    stop_words = {"的", "了", "吗", "呢", "是", "有", "在", "被", "把", "和", "与",
                  "还", "也", "都", "就", "会", "能", "要", "可以", "什么", "怎么",
                  "哪些", "多少", "如何", "是否", "关于", "最近", "目前", "请问",
                  "告诉", "一下", "这个", "那个", "哪个"}

    keywords = []
    # 先按标点和空格拆
    segments = re.split(r"[，。？！\s,.\?!、：:；;]+", query)
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        # 去掉停用词
        for sw in stop_words:
            seg = seg.replace(sw, " ")
        # 按空格拆出子片段
        for w in seg.split():
            w = w.strip()
            if len(w) >= 2:
                keywords.append(w)

    # 对长关键词做滑动窗口切分（2-4字符），提升中文匹配率
    extra = []
    for kw in keywords:
        if len(kw) > 4:
            # 切成2字符和4字符的片段
            for size in [2, 3, 4]:
                for i in range(len(kw) - size + 1):
                    sub = kw[i:i + size]
                    if sub not in stop_words and sub not in keywords:
                        extra.append(sub)

    # 原始关键词在前，子片段在后，去重
    seen = set()
    result = []
    for kw in keywords + extra:
        if kw not in seen:
            seen.add(kw)
            result.append(kw)
    return result


def _keyword_search(keywords: list[str], top_k: int) -> list[dict]:
    """使用 ChromaDB 的 where_document 逐个关键词匹配，合并去重。"""
    collection = get_collection()
    if collection.count() == 0 or not keywords:
        return []

    seen_ids = set()
    sources = []

    # 逐个关键词搜索（ChromaDB where_document 不支持 $or）
    for kw in keywords[:8]:
        try:
            results = collection.get(
                where_document={"$contains": kw},
                include=["documents", "metadatas"],
                limit=top_k,
            )
            for i in range(len(results["ids"])):
                eid = results["ids"][i]
                if eid in seen_ids:
                    continue
                seen_ids.add(eid)
                metadata = results["metadatas"][i]
                sources.append({
                    "email_id": metadata["email_id"],
                    "subject": metadata["subject"],
                    "from_name": metadata["from_name"],
                    "from_email": metadata["from_email"],
                    "date": metadata["date"],
                    "tags": metadata.get("tags", ""),
                    "attachments": metadata.get("attachments", ""),
                    "document": results["documents"][i],
                    "distance": 0.5,
                })
        except Exception:
            continue

    return sources[:top_k]


def _semantic_search(query: str, top_k: int) -> list[dict]:
    """使用 ChromaDB 的向量检索。"""
    collection = get_collection()
    if collection.count() == 0:
        return []

    results = collection.query(
        query_texts=[query],
        n_results=min(top_k, collection.count()),
    )

    sources = []
    for i in range(len(results["ids"][0])):
        metadata = results["metadatas"][0][i]
        sources.append({
            "email_id": metadata["email_id"],
            "subject": metadata["subject"],
            "from_name": metadata["from_name"],
            "from_email": metadata["from_email"],
            "date": metadata["date"],
            "tags": metadata.get("tags", ""),
            "attachments": metadata.get("attachments", ""),
            "document": results["documents"][0][i],
            "distance": results["distances"][0][i] if results.get("distances") else None,
        })
    return sources


def retrieve(query: str, top_k: int | None = None) -> list[dict]:
    """混合检索：关键词匹配 + 语义检索，去重合并。"""
    top_k = top_k or RETRIEVAL_TOP_K

    # 1. 关键词检索
    keywords = _extract_keywords(query)
    keyword_results = _keyword_search(keywords, top_k)

    # 2. 语义检索
    semantic_results = _semantic_search(query, top_k)

    # 3. 合并去重，关键词匹配的排在前面
    seen_ids = set()
    merged = []

    # 先放关键词匹配的结果
    for r in keyword_results:
        if r["email_id"] not in seen_ids:
            seen_ids.add(r["email_id"])
            merged.append(r)

    # 再放语义检索的结果
    for r in semantic_results:
        if r["email_id"] not in seen_ids:
            seen_ids.add(r["email_id"])
            merged.append(r)

    return merged[:top_k]
