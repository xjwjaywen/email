# 邮件检索问答系统

基于 AI 的邮件智能检索系统。通过自然语言提问，从历史邮件（含附件）中检索信息，生成回答并附上来源。

## 功能特性

- **智能问答**：自然语言提问，AI 从邮件库中检索并生成答案
- **来源引用**：每个回答附带邮件来源（主题、发件人、时间、附件）
- **多对话管理**：创建多个独立对话，随时切换
- **上下文记忆**：支持多轮对话，理解上下文
- **Memory 系统**：记住用户偏好，跨对话持久化
- **附件检索**：支持 PDF、Word、Excel、PPT、CSV、TXT 等格式的附件内容检索

## 技术栈

- **前端**：React + TypeScript + Tailwind CSS
- **后端**：Python + FastAPI
- **向量数据库**：ChromaDB
- **LLM**：OpenAI GPT-4o
- **Embedding**：OpenAI text-embedding-3-small
- **元数据存储**：SQLite

## 快速开始

### 环境要求

- Python 3.11+
- Node.js 18+
- OpenAI API Key

### 后端启动

```bash
cd backend
pip install -r requirements.txt
# 生成假邮件数据
python seed_data.py
# 启动服务
uvicorn app.main:app --reload --port 8000
```

### 前端启动

```bash
cd frontend
npm install
npm run dev
```

### 环境变量

```bash
# backend/.env
OPENAI_API_KEY=your_api_key_here
```

## 项目结构

```
email/
├── backend/           # Python 后端
│   ├── app/           # 应用代码
│   ├── data/          # 邮件数据和附件
│   ├── tests/         # 后端测试
│   └── seed_data.py   # 数据生成脚本
├── frontend/          # React 前端
│   └── src/           # 前端源码
├── REQUIREMENTS.md    # 需求文档
├── TEST.md            # 测试指南
└── TODO.md            # 开发任务清单
```

## 文档

- [需求文档](REQUIREMENTS.md) — 完整功能需求和技术设计
- [测试指南](TEST.md) — 测试用例和运行方法
- [开发任务](TODO.md) — 开发进度追踪
