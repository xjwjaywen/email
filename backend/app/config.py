import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
EMAILS_DIR = DATA_DIR / "emails"
ATTACHMENTS_DIR = DATA_DIR / "attachments"
DB_PATH = DATA_DIR / "app.db"
CHROMA_DIR = DATA_DIR / "chroma"

# MiniMax API 配置
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY", "")
MINIMAX_BASE_URL = "https://api.minimax.chat/v1"
MINIMAX_CHAT_MODEL = "MiniMax-M2.7"

# OpenAI API 配置（备用）
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_CHAT_MODEL = "gpt-4o"

RETRIEVAL_TOP_K = 5

# 使用 MiniMax 还是 OpenAI
USE_MINIMAX = bool(MINIMAX_API_KEY)
