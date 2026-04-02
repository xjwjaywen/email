# 开发任务清单

## M1 — 数据层

- [ ] 初始化后端项目结构 (FastAPI + requirements.txt)
- [ ] 编写假邮件数据生成脚本 `seed_data.py`
  - [ ] 定义邮件 JSON 结构
  - [ ] 生成 30+ 封邮件，覆盖 8 个业务场景
  - [ ] 生成各格式附件文件 (pdf/docx/xlsx/pptx/txt/csv/png/jpg/zip)
  - [ ] 确保附件有实际可解析内容
- [ ] 编写文件解析工具 `file_parser.py`
  - [ ] PDF 解析
  - [ ] DOCX 解析
  - [ ] XLSX 解析
  - [ ] PPTX 解析
  - [ ] CSV 解析
  - [ ] TXT 解析
- [ ] 编写数据层测试 (D-01 ~ D-04, P-01 ~ P-06)

## M2 — 检索引擎

- [ ] 编写邮件数据加载器 `email_loader.py`
- [ ] 编写索引服务 `indexer.py`
  - [ ] 邮件正文/主题 Embedding
  - [ ] 附件内容 Embedding
  - [ ] 存入 ChromaDB
- [ ] 编写检索服务 `retriever.py`
  - [ ] 语义检索，返回 Top-K 相关邮件
  - [ ] 结果包含来源信息
- [ ] 编写 LLM 服务 `llm.py`
  - [ ] 构建 RAG prompt
  - [ ] 调用 OpenAI 生成答案
  - [ ] 解析并返回带来源的回答
- [ ] 编写检索测试 (R-01 ~ R-04)

## M3 — 对话管理

- [ ] 设计 SQLite 数据库 schema (对话、消息、Memory)
- [ ] 编写数据模型 (email.py, conversation.py, memory.py)
- [ ] 编写对话管理 API (conversations.py router)
  - [ ] CRUD 对话
  - [ ] 消息历史查询
- [ ] 编写聊天 API (chat.py router)
  - [ ] 接收用户消息
  - [ ] 调用检索+LLM
  - [ ] 上下文拼接（多轮对话）
  - [ ] 返回答案+来源
- [ ] 编写 Memory 服务和 API
  - [ ] 自动从对话中提取 Memory
  - [ ] Memory 查询和删除
  - [ ] Memory 融入检索流程
- [ ] 编写对话管理测试 (C-01 ~ C-07, M-01 ~ M-04)

## M4 — 前端界面

- [ ] 初始化前端项目 (React + TypeScript + Tailwind)
- [ ] 编写 API 服务层 (services/api.ts)
- [ ] 编写类型定义 (types/)
- [ ] 编写侧边栏组件 (Sidebar.tsx)
  - [ ] 对话列表
  - [ ] 新建对话按钮
  - [ ] 对话右键菜单（重命名、删除）
- [ ] 编写聊天区域组件 (ChatArea.tsx)
  - [ ] 消息气泡 (MessageBubble.tsx)
  - [ ] 来源引用卡片 (SourceCard.tsx)
  - [ ] 附件预览
- [ ] 编写输入区域组件 (InputArea.tsx)
  - [ ] Enter 发送 / Shift+Enter 换行
  - [ ] Loading 状态
- [ ] 整合 App.tsx 布局
- [ ] 响应式设计适配
- [ ] 编写前端测试 (F-01 ~ F-06)

## M5 — 集成测试与文档

- [ ] 端到端测试 (E-01 ~ E-03)
- [ ] 更新 README.md（最终版）
- [ ] 更新 CLAUDE.md（架构信息）
- [ ] 更新 TEST.md（补充遗漏用例）
- [ ] 编写 PROGRESS.md（经验总结）
