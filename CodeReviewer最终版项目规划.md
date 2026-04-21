# CodeReviewer 最终版：隐私优先的测试缺口检测 Agent

## 一、项目定位（修正后）

### 一句话描述
一个连接 GitHub 的 AI 代码评审 Agent，**专注检测 PR 中的测试覆盖缺口**，支持 Claude Sonnet（云端）和自训 Qwen2.5-Coder-7B（自托管）双模式，使用 Bug-fix Mining 构建客观评测集。

### 三个差异化锚点

**1. 垂直聚焦：只做测试缺口检测**
不做通用代码评审（那是 CodeRabbit/PR-Agent 的赛道）。只聚焦一个问题：**PR 改了业务逻辑，但没有对应的测试变更或新增测试**。这是 code review 中最常被忽略、却最高频导致线上 bug 的问题。

为什么选这个切角：
- 检测标准客观（改了 `src/` 但没动 `tests/`，可以程序化判断）
- 每个开发者都理解价值（"你改了逻辑但没改测试"是最常见的 review 意见）
- 和 Bug-fix Mining eval 天然契合（很多 bug 就是因为缺测试）
- CodeRabbit/PR-Agent 没有专门做这件事

具体检测能力：
- **测试文件缺失**：业务代码改了但对应的测试文件完全没变
- **测试场景缺失**：虽然测试文件改了，但新增/修改的代码路径没有被覆盖
- **边界条件遗漏**：改了输入校验/错误处理逻辑，但测试只覆盖了 happy path
- **测试用例建议**：不只是指出缺什么测试，还给出具体的测试场景建议

**2. 隐私可控：云端 + 自托管双模式**
- 模式 A（云端）：用 Claude Sonnet / DeepSeek，适合开源项目和个人开发者
- 模式 B（自托管）：用你在 L20 上 LoRA 微调的 Qwen2.5-Coder-7B，适合企业私有代码

面试叙事："企业核心代码不敢发给第三方 API。我训了一个 7B 模型专做测试缺口检测，企业在自己的 GPU 上跑，代码不出内网。"

**3. Bug-fix Mining Eval：客观的评测方法**
不用"人工 review 评论"当 ground truth（主观、有噪声），而是从 GitHub 上挖掘"PR A 引入了 bug → 后续 PR B 修复了 bug 并补了测试"的配对。把 PR A 丢给 agent，看它能不能指出缺少的测试。Ground truth 是客观的——bug 确实存在，测试确实缺失。

---

## 二、系统架构

```
┌──────────────┐     Webhook      ┌─────────────────────────────┐
│   GitHub     │ ──────────────→  │   API Server (FastAPI)      │
│   (PR event) │                  │                             │
└──────────────┘                  │  ┌───────────────────────┐  │
                                  │  │  Test Gap Detector     │  │
       GitHub API ←───────────    │  │  (核心检测引擎)        │  │
       (post comments)            │  │                       │  │
                                  │  │  ┌─────────────────┐  │  │
                                  │  │  │ Static Analyzer  │  │  │
                                  │  │  │ (AST/import分析) │  │  │
                                  │  │  └─────────────────┘  │  │
                                  │  │  ┌─────────────────┐  │  │
                                  │  │  │ LLM Reviewer    │  │  │
                                  │  │  │ (语义级分析)     │  │  │
                                  │  │  └─────────────────┘  │  │
                                  │  └───────────────────────┘  │
                                  │                             │
                                  │  ┌───────────────────────┐  │
                                  │  │  Model Router         │  │
                                  │  │  Claude / DeepSeek /  │  │
                                  │  │  Qwen-7B-SFT (本地)   │  │
                                  │  └───────────────────────┘  │
                                  │                             │
                                  │  ┌───────────────────────┐  │
                                  │  │  Trace + Cost Logger  │  │
                                  │  └───────────────────────┘  │
                                  └─────────────────────────────┘
```

### 技术栈

| 组件 | 选型 | 理由 |
|------|------|------|
| Web 框架 | FastAPI | 原生异步，自动 API 文档 |
| LLM 调用 | httpx（直接调 API） | 不用 LangChain，保持轻量 |
| 云端模型 | Claude Sonnet / DeepSeek-V3 | 主力模型，高质量 |
| 自托管模型 | Qwen2.5-Coder-7B + LoRA | L20 48G 可轻松承载 |
| 训练框架 | transformers + PEFT + TRL (SFTTrainer) | 社区最成熟的 LoRA SFT 方案 |
| 推理服务 | vLLM | 高效推理 serving，L20 兼容 |
| AST 解析 | Python ast + tree-sitter（多语言） | 静态分析用 |
| 数据库 | SQLite | 轻量，够用 |
| 部署 | Docker + Railway（云端）/ 本地部署（自托管模式） | 灵活 |
| GitHub 集成 | PyGithub + GitHub App API | 官方方式 |

---

## 三、核心检测引擎详细设计

测试缺口检测分两层：**静态分析层**（程序化规则，快速）+ **LLM 语义层**（深度理解，精准）。

### 3.1 静态分析层（无需 LLM，毫秒级）

```python
class StaticTestGapAnalyzer:
    """
    基于规则的测试缺口检测。不调 LLM，纯代码分析。
    
    这一层的作用：
    1. 快速筛选出"高度可疑"的文件，减少 LLM 调用量
    2. 提供结构化信息给 LLM 层，提升 LLM 分析质量
    3. 对于明显的缺口（改了 src 没动 test），直接输出结论
    """
    
    def analyze(self, pr: PullRequest) -> StaticAnalysisResult:
        changed_files = self.categorize_files(pr.files)
        # 1. 文件映射：找到每个业务文件对应的测试文件
        # 2. 变更匹配：业务文件改了，对应测试文件改没改？
        # 3. 函数级分析：新增/修改了哪些函数？测试文件里有没有对应测试？
        return result
```

**文件映射策略：**
```python
class TestFileMapper:
    """
    找到业务文件对应的测试文件。
    
    常见的项目结构模式：
    
    模式 1：镜像目录
    src/auth/login.py  →  tests/auth/test_login.py
    
    模式 2：同目录
    app/models.py  →  app/test_models.py
    
    模式 3：Django 风格
    myapp/views.py  →  myapp/tests/test_views.py
    
    模式 4：前端项目
    src/components/Button.tsx  →  src/components/Button.test.tsx
                               或 src/components/__tests__/Button.test.tsx
    
    策略：
    1. 先检查项目是否有 pytest.ini / setup.cfg / jest.config 等配置
    2. 按上述模式依次尝试匹配
    3. 如果都不匹配，用文件名相似度做模糊匹配
    4. 缓存匹配结果，同一仓库不重复计算
    """
    
    def find_test_file(self, source_file: str, repo_structure: RepoStructure) -> Optional[str]:
        pass
```

**函数级变更检测：**
```python
class FunctionChangeDetector:
    """
    用 AST 解析 diff 前后的文件，识别哪些函数被新增/修改/删除。
    
    输出示例：
    {
        "added": ["validate_token", "refresh_session"],
        "modified": ["authenticate_user"],  # 函数体有变化
        "deleted": ["legacy_login"],
        "signature_changed": ["create_user"]  # 参数变了
    }
    
    这个信息会传给 LLM 层，让它精确判断哪些函数需要测试。
    """
    
    def detect_changes(self, old_content: str, new_content: str) -> FunctionChanges:
        old_ast = ast.parse(old_content)
        new_ast = ast.parse(new_content)
        # 对比两棵 AST 的函数定义
        pass
```

### 3.2 LLM 语义层（深度分析）

静态分析能找到"改了 src 没动 test"的明显缺口，但有些缺口需要语义理解才能发现：
- 改了错误处理逻辑，但测试只覆盖了正常路径
- 改了数据校验规则，但没有测试边界值
- 重构了函数实现但语义变了，旧测试还能过但实际上覆盖不全

```python
class LLMTestGapReviewer:
    """
    用 LLM 做深度的测试缺口分析。
    
    输入：
    - 静态分析的结果（哪些函数改了、对应测试文件状态）
    - 改动代码的完整上下文
    - 现有测试代码
    
    输出：
    - 缺失的测试场景列表
    - 每个场景的具体测试建议（伪代码级别）
    - 风险等级评估
    """
    
    async def review(
        self,
        static_result: StaticAnalysisResult,
        code_context: CodeContext,
        model: str = "claude-sonnet"  # 或 "qwen-7b-sft"
    ) -> list[TestGapFinding]:
        prompt = self.build_prompt(static_result, code_context)
        response = await self.llm_client.generate(prompt, model=model)
        findings = self.parse_response(response)
        return findings
```

**Prompt 模板（核心）：**
```
你是一位专注于测试覆盖率的代码审查专家。

## 任务
分析以下 PR 改动，找出测试覆盖的缺口。

## PR 改动摘要
{pr_summary}

## 被修改的函数
{function_changes}

## 改动代码
```{language}
{changed_code}
```

## 现有测试代码
```{language}
{existing_tests}
```

## 静态分析已发现的问题
{static_findings}

## 请分析：
1. 新增/修改的代码路径中，哪些没有被现有测试覆盖？
2. 对于每个未覆盖的路径，建议补充什么测试？
3. 是否有边界条件或异常路径需要额外测试？

## 输出格式（JSON）
[
  {
    "function": "被检测的函数名",
    "gap_type": "missing_test | incomplete_coverage | missing_edge_case",
    "description": "具体缺什么测试",
    "risk_level": "high | medium | low",
    "suggested_test": "建议的测试伪代码（pytest 风格）",
    "confidence": 0.0-1.0
  }
]

如果没有发现测试缺口，输出空数组 []。
不要输出已经被测试覆盖的内容。只输出缺失的部分。
```

### 3.3 两层协作

```python
class TestGapDetector:
    """
    两层检测的协作逻辑：
    
    1. 静态分析层先跑，快速标记出：
       - 确定有缺口的（改了 src，测试文件不存在）→ 直接输出，不用 LLM
       - 可能有缺口的（测试文件存在但不确定覆盖是否充分）→ 传给 LLM 层
       - 确定没问题的（只改了测试文件 / 只改了配置）→ 跳过
    
    2. LLM 层只处理"可能有缺口"的部分，节省 token
    
    好处：
    - 对于明显的缺口，0 token 消耗
    - LLM 收到的上下文更精准（有静态分析的结构化信息辅助）
    - 总成本降低 40-60%
    """
    
    async def detect(self, pr: PullRequest) -> list[TestGapFinding]:
        # Step 1: 静态分析
        static_result = self.static_analyzer.analyze(pr)
        
        # Step 2: 确定性缺口直接输出
        definite_gaps = static_result.definite_gaps
        
        # Step 3: 可能的缺口发给 LLM 深度分析
        if static_result.suspicious_files:
            llm_gaps = await self.llm_reviewer.review(
                static_result, 
                self.build_context(pr, static_result.suspicious_files)
            )
        else:
            llm_gaps = []
        
        # Step 4: 合并、去重、排序
        all_gaps = self.merge_and_rank(definite_gaps + llm_gaps)
        return all_gaps
```

---

## 四、Bug-fix Mining Eval 详细设计

### 4.1 数据挖掘流程

```
Step 1: 选定目标仓库
        - 选 10-15 个主流开源项目（Django, Flask, FastAPI, requests,
          scikit-learn, pandas, httpx, pytest, black, mypy 等）
        - 要求：有良好的测试习惯，PR 流程规范

Step 2: 挖掘 Bug-fix 配对
        对每个仓库，用 GitHub API + git log 筛选：
        
        a. 找到所有 commit message 包含 "fix"/"bug"/"patch"/"resolve" 的 commit
        b. 检查这些 commit 是否同时修改了 src 文件和 test 文件
        c. 如果是，说明这个 fix 补了测试 → 反推之前的代码有测试缺口
        d. 用 git blame 追溯是哪个 PR 引入了有问题的代码
        e. 得到配对：(PR_buggy, commit_fix)

Step 3: 验证配对质量
        对每个配对，检查：
        - fix commit 新增的测试是否能暴露原始 bug（回退到 buggy 版本跑测试）
        - 测试失败是否和 fix 的改动直接相关
        - 过滤掉：纯 typo 修复、依赖版本更新、文档修改

Step 4: 构建评测集
        每个 case 包含：
        {
            "repo": "django/django",
            "buggy_pr": {
                "diff": "...",           # 引入 bug 的 PR diff
                "files_changed": [...],
                "description": "..."
            },
            "fix_commit": {
                "diff": "...",           # 修复 bug 的 commit diff
                "added_tests": [...],    # 新增的测试代码
                "description": "..."
            },
            "expected_gaps": [           # ground truth: 应该被检测到的缺口
                {
                    "file": "src/auth.py",
                    "function": "validate_token",
                    "gap_type": "missing_edge_case",
                    "description": "缺少 token 过期场景的测试"
                }
            ]
        }

目标：构建 50-80 个高质量 case
```

### 4.2 评测指标

```python
class EvalMetrics:
    """
    三个核心指标：
    
    1. Detection Rate（检测率）
       Agent 检出的缺口 / Ground truth 中的总缺口数
       "该发现的问题，发现了多少？"
    
    2. Precision（精确率）
       Agent 检出的真正缺口 / Agent 检出的总数
       "Agent 说的问题，有多少是真的？"
    
    3. Suggestion Quality（建议质量）
       Agent 建议的测试场景 vs fix commit 中实际补充的测试
       用 BLEU / ROUGE 或人工评分衡量建议的相关性
    
    辅助指标：
    - 成本：每个 PR 的平均 token 消耗和美元成本
    - 延迟：每个 PR 的平均处理时间
    - 静态分析贡献率：有多少缺口是纯静态分析发现的（不需要 LLM）
    """
```

### 4.3 Eval Pipeline（跑在 CI 里）

```yaml
# .github/workflows/eval.yml
name: Eval Pipeline
on:
  push:
    paths:
      - 'server/review/**'    # 评审逻辑改了
      - 'prompts/**'          # prompt 改了
      - 'eval/**'             # eval 代码改了

jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run eval
        run: python eval/run_eval.py --dataset eval/dataset/cases.json
      - name: Compare with baseline
        run: python eval/compare.py --current eval/results/latest.json --baseline eval/results/baseline.json
      - name: Post results
        run: python eval/report.py --output eval/reports/
```

---

## 五、SFT 微调详细方案

### 5.1 学习路线（你的第一次微调）

**Week 1：理论 + 环境准备（不急着训，先理解）**

学习内容：
- LoRA 原理（2小时）：理解低秩分解的思路，为什么只训很少的参数就能适配新任务
- SFTTrainer 使用方法（3小时）：HuggingFace TRL 库的 SFTTrainer，它封装了训练循环
- 数据格式（2小时）：理解 instruction-following 数据集的 JSON 格式
- 推荐阅读：HuggingFace PEFT 文档、TRL 文档、阿里云 Qwen2.5-Coder 微调教程

环境搭建：
```bash
# L20 48G 足够跑 LoRA SFT
pip install torch transformers peft trl datasets accelerate
pip install vllm  # 用于推理

# 验证 GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"
# 应该输出 NVIDIA L20 或类似

# 验证显存
python -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')"
# 应该输出 ~48.0 GB
```

**Week 2：数据准备**

这是最重要的一步。数据质量 > 数据量 > 训练技巧。

训练数据来源：
1. 从你的 Bug-fix Mining eval 集扩展（已有结构化标注）
2. 从知名开源项目爬取含有测试相关评论的 PR review
3. 手工构造一批高质量的 "代码改动 + 测试缺口分析" 示例

数据格式：
```json
{
  "instruction": "分析以下 PR 改动，找出测试覆盖的缺口。\n\n## 改动代码\n```python\ndef validate_token(token: str) -> bool:\n    if not token:\n        raise ValueError('Token is required')\n+   if len(token) > 1024:\n+       raise ValueError('Token too long')\n    return jwt.decode(token, SECRET_KEY)\n```\n\n## 现有测试\n```python\ndef test_validate_token_success():\n    assert validate_token('valid.jwt.token') == True\n\ndef test_validate_token_empty():\n    with pytest.raises(ValueError):\n        validate_token('')\n```",
  
  "output": "[{\"function\": \"validate_token\", \"gap_type\": \"missing_edge_case\", \"description\": \"新增了 token 长度校验(>1024)，但没有对应的测试用例\", \"risk_level\": \"medium\", \"suggested_test\": \"def test_validate_token_too_long():\\n    with pytest.raises(ValueError, match='Token too long'):\\n        validate_token('a' * 1025)\", \"confidence\": 0.95}]"
}
```

目标数据量：
- 高质量样本 800-1500 条
- 其中 70% 来自真实 PR 数据，30% 手工构造（覆盖各种 gap_type）

**Week 3：训练**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

# 模型加载
model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # L20 支持 bf16
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# LoRA 配置
lora_config = LoraConfig(
    r=16,                    # rank，16 是常用值
    lora_alpha=32,           # scaling factor
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 注意力层
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# 训练配置
training_config = SFTConfig(
    output_dir="./checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=4,    # L20 48G 可以跑 batch=4
    gradient_accumulation_steps=4,     # 有效 batch size = 16
    learning_rate=2e-4,
    warmup_steps=100,
    logging_steps=10,
    save_steps=200,
    max_seq_length=4096,              # 代码 review 需要较长上下文
    bf16=True,                         # L20 支持
)

# 开始训练
trainer = SFTTrainer(
    model=model,
    args=training_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=lora_config,
)
trainer.train()

# 保存 LoRA adapter（很小，几十 MB）
trainer.save_model("./qwen-7b-test-gap-reviewer")
```

训练预计：
- 1000 条数据，3 个 epoch，约 2-4 小时（L20 上）
- 显存占用预计 20-25 GB（LoRA + bf16），远低于 48GB 上限
- 可以多跑几轮实验，调 rank、learning rate、epoch 数

### 5.2 推理部署

```python
# 用 vLLM 部署微调后的模型
# vLLM 支持 LoRA adapter 的动态加载

from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen2.5-Coder-7B-Instruct",
    enable_lora=True,
    max_lora_rank=16,
)

# 加载你的 LoRA adapter
sampling_params = SamplingParams(temperature=0.3, max_tokens=2048)
output = llm.generate(
    prompts=[review_prompt],
    sampling_params=sampling_params,
    lora_request=LoRARequest("test-gap-reviewer", 1, "./qwen-7b-test-gap-reviewer")
)
```

### 5.3 模型对比实验

在 Bug-fix Mining eval 集上对比三个模型：

| 模型 | 预期 Detection Rate | 预期 Precision | 单次成本 | 部署方式 |
|------|---------------------|---------------|---------|---------|
| Claude Sonnet | ~65-75% | ~70-80% | ~$0.08 | 云端 API |
| DeepSeek-V3 | ~55-65% | ~60-70% | ~$0.02 | 云端 API |
| Qwen-7B-SFT (你的) | ~50-60% | ~55-65% | ~$0.001 | 本地 L20 |

面试叙事的关键数据点：
- "我的 7B 模型达到 Claude Sonnet 约 75-80% 的检测质量"
- "但推理成本是 Claude 的 1/80"
- "企业自托管场景下，代码不出内网"

如果 SFT 模型效果不好（低于基础模型），也是有价值的发现：
- 分析为什么效果不好（数据量不够？数据质量问题？任务太复杂？）
- 尝试改进（增加数据、调整数据配比、换 14B 模型）
- 诚实报告在面试时反而加分，展示你的科学态度

---

## 六、工程能力展示点（保留 + 精简）

从上一版保留核心工程点，去掉 Dashboard 相关内容：

### 6.1 异步任务处理
（同上一版，Webhook 10秒响应要求 → 后台异步处理）

### 6.2 成本控制
```
三层策略：
1. 静态分析前置：明显的缺口不调 LLM，0 token
2. 上下文精准投递：只发相关代码给 LLM，不发整个 PR diff
3. 预算上限：每个 PR 设 token 上限，超限只处理高优先级文件
```

### 6.3 容错
```
三层设计：
1. 单文件隔离：一个文件分析失败不影响其他文件
2. JSON 解析多级 fallback（你在 Darwinian 做过）
3. 模型降级：Claude 不可用 → DeepSeek → 本地 7B → 纯静态分析
```

### 6.4 Trace 系统
```
每次评审的完整执行轨迹：
[0] webhook_received     | 5ms    | $0.00
[1] fetch_pr_diff        | 350ms  | $0.00
[2] static_analysis      | 120ms  | $0.00  | found: 2 definite gaps
[3] llm_review:auth.py   | 2800ms | $0.03  | model: claude-sonnet | found: 1 gap
[4] llm_review:api.py    | 3100ms | $0.04  | model: claude-sonnet | found: 0 gaps
[5] post_comments        | 600ms  | $0.00  | posted: 3 comments

Total: 7.0s | $0.07 | 3 test gaps found
```

### 6.5 幂等性
```python
async def handle_webhook(payload):
    # GitHub 可能重复触发 webhook（网络问题时自动重试）
    idempotency_key = f"{payload['pull_request']['id']}:{payload['pull_request']['head']['sha']}"
    
    if await db.review_exists(idempotency_key):
        return Response(status_code=200)  # 已处理过，跳过
    
    await db.create_review_record(idempotency_key, status="pending")
    background_tasks.add_task(process_review, payload, idempotency_key)
    return Response(status_code=202)
```

### 6.6 安全
```
- Webhook 签名验证（HMAC-SHA256）
- GitHub App token 定期轮换
- 本地模式下代码不出内网
- Dashboard API 用 GitHub OAuth 认证
- 速率限制：每个仓库每小时最多 20 次评审
```

### 6.7 输出格式（极简 HTML 报告替代 Dashboard）

不做完整的 Vue 前端。每次评审自动生成一个静态 HTML 报告页：
```python
def generate_report(review_result: ReviewResult) -> str:
    """
    生成一个自包含的 HTML 报告。
    包含：评审结果摘要、每个 finding 的详情、trace 时间线、成本明细。
    用 Jinja2 模板渲染，不需要前端框架。
    """
```

报告页可以通过 API 访问：`GET /reviews/{review_id}/report`

---

## 七、项目结构（修正后）

```
code-reviewer/
├── README.md
├── pyproject.toml
├── Dockerfile
├── .github/workflows/
│   ├── ci.yml                    # 测试 + lint
│   └── eval.yml                  # 自动 eval
│
├── server/                       # 后端
│   ├── app.py                    # FastAPI 入口
│   ├── config.py                 # 配置（env vars）
│   │
│   ├── webhook/                  # GitHub Webhook
│   │   ├── handler.py
│   │   ├── signature.py          # 签名验证
│   │   └── idempotency.py        # 幂等性控制
│   │
│   ├── github/                   # GitHub API
│   │   ├── client.py
│   │   ├── diff_parser.py
│   │   └── comment_poster.py     # inline comment 位置计算
│   │
│   ├── detection/                # 核心检测引擎
│   │   ├── detector.py           # 两层检测协调
│   │   ├── static_analyzer.py    # 静态分析层
│   │   ├── test_file_mapper.py   # 业务文件→测试文件映射
│   │   ├── function_differ.py    # AST 级别函数变更检测
│   │   └── llm_reviewer.py       # LLM 语义分析层
│   │
│   ├── llm/                      # LLM 调用
│   │   ├── client.py             # 多模型统一接口
│   │   ├── router.py             # 模型路由（云端/本地）
│   │   ├── prompts.py            # Prompt 模板
│   │   └── parser.py             # 输出解析 + fallback
│   │
│   ├── trace/                    # 可观测性
│   │   ├── logger.py
│   │   ├── cost.py
│   │   └── report.py             # HTML 报告生成
│   │
│   └── db/
│       ├── models.py
│       └── repository.py
│
├── training/                     # SFT 微调
│   ├── README.md                 # 训练文档
│   ├── prepare_data.py           # 训练数据准备
│   ├── train.py                  # LoRA SFT 训练脚本
│   ├── evaluate_model.py         # 微调模型评估
│   ├── serve.py                  # vLLM 推理服务启动
│   └── configs/
│       └── lora_config.yaml      # 训练超参数
│
├── eval/                         # 评测
│   ├── mine_bugfix_pairs.py      # Bug-fix 配对挖掘
│   ├── build_dataset.py          # 评测集构建
│   ├── run_eval.py               # 运行评测
│   ├── compare.py                # 对比分析
│   ├── report.py                 # 生成 eval 报告
│   └── dataset/
│       └── cases.json
│
├── prompts/
│   ├── test_gap_review.j2
│   └── test_suggestion.j2
│
└── tests/                        # 单元测试（覆盖率 >70%）
    ├── test_diff_parser.py
    ├── test_test_file_mapper.py
    ├── test_function_differ.py
    ├── test_static_analyzer.py
    ├── test_idempotency.py
    └── fixtures/
```

---

## 八、13 周实现路线图

### Phase 1：基础设施（Week 1-3）

**Week 1：GitHub App + Webhook**
- [ ] 注册 GitHub App，获取 credentials
- [ ] FastAPI 项目骨架搭建
- [ ] Webhook 接收 + 签名验证
- [ ] GitHub API 封装（获取 PR、diff、发评论）
- [ ] 用 ngrok 本地调试 webhook 链路
- [ ] 幂等性控制实现

**Week 2：Diff 解析 + 基础检测**
- [ ] unified diff 解析器（处理 rename、binary、submodule 等 edge case）
- [ ] inline comment 位置计算（diff line → file line 的映射，这是个坑，预留时间）
- [ ] 测试文件映射器（4 种常见项目结构模式）
- [ ] 端到端打通：PR 触发 → 简单检测 → 发评论

**Week 3：静态分析层**
- [ ] AST 解析器（Python 用 ast 模块，JS/TS 用 tree-sitter）
- [ ] 函数级变更检测器
- [ ] 静态测试缺口检测（改了 src 没动 test 的明显缺口）
- [ ] 在 5 个真实 PR 上测试静态分析效果

**Week 3 检查点：** 静态分析能正确检测"改了业务代码但没改测试文件"的基本缺口

### Phase 2：核心检测能力（Week 4-6）

**Week 4：LLM 语义分析层**
- [ ] LLM 调用封装（先接 DeepSeek，便宜，适合调试）
- [ ] Prompt 模板设计和迭代
- [ ] JSON 输出解析 + 多级 fallback
- [ ] 两层检测协调逻辑（静态前置 + LLM 深度）

**Week 5：工程打磨**
- [ ] 异步任务处理（后台 worker）
- [ ] Trace 系统实现
- [ ] 成本计算和预算控制
- [ ] 容错：单文件隔离 + 模型降级
- [ ] 接入 Claude Sonnet 作为第二个模型

**Week 6：质量调优 + 基础评测**
- [ ] 在 15-20 个真实 PR 上跑通完整流程
- [ ] 人工审核输出质量，迭代 prompt
- [ ] 评论数量和质量的平衡调优
- [ ] HTML 报告生成器

**Week 6 检查点：** 完整系统可用，在真实 PR 上能输出有价值的测试缺口检测结果

### Phase 3：Bug-fix Mining Eval（Week 7-8）

**Week 7：数据挖掘**
- [ ] 编写 GitHub PR/commit 数据爬取脚本
- [ ] 实现 bug-fix 配对挖掘逻辑
- [ ] 从 10 个开源项目中挖掘候选配对
- [ ] 人工验证和清洗，筛选 50-80 个高质量 case

**Week 8：Eval Pipeline**
- [ ] 构建标准化评测数据集（JSON 格式）
- [ ] 实现评测脚本（Detection Rate, Precision, Suggestion Quality）
- [ ] 在 eval 集上运行 Claude Sonnet 和 DeepSeek 的基线评测
- [ ] 设置 GitHub Actions 自动 eval

**Week 8 检查点：** 有 50+ case 的评测集，有基线数据，eval pipeline 自动化

### Phase 4：SFT 微调（Week 9-11）

**Week 9：学习 + 数据准备**
- [ ] 学习 LoRA 原理和 PEFT 库使用（2-3 天）
- [ ] 跑通一个最简的 LoRA SFT 示例（用公开数据集，验证环境）
- [ ] 从 eval 集 + 额外爬取的 PR review 数据构建训练集
- [ ] 数据清洗和格式化（instruction-output JSON 格式）
- [ ] 目标：800-1500 条高质量训练样本

**Week 10：训练 + 迭代**
- [ ] 第一轮训练（3 epoch, rank=16, lr=2e-4）
- [ ] 在 eval 集上评估效果
- [ ] 分析错误案例：模型哪里做得不好？
- [ ] 调整超参数（rank/lr/epoch）再训 1-2 轮
- [ ] 尝试不同的数据配比（更多/更少手工数据）

**Week 11：部署 + 对比实验**
- [ ] 用 vLLM 部署微调后的模型
- [ ] 集成到主系统作为本地模式
- [ ] 跑三模型对比实验（Claude Sonnet / DeepSeek / Qwen-7B-SFT）
- [ ] 生成对比报告（质量 vs 成本的 trade-off 分析）

**Week 11 检查点：** SFT 模型跑通，对比数据出来

### Phase 5：收尾（Week 12-13）

**Week 12：部署 + 测试**
- [ ] Dockerfile 编写
- [ ] 部署到 Railway / Fly.io
- [ ] 配置域名 + HTTPS
- [ ] 单元测试补全（覆盖率 >70%）
- [ ] 准备演示仓库（几个示例 PR）

**Week 13：文档 + 面试准备**
- [ ] 写 README（安装指南 + 架构图 + 评测结果 + 设计决策）
- [ ] 写技术博客（一篇，讲清楚核心故事）
- [ ] 录制演示视频（给不方便安装的面试官看）
- [ ] 消融实验（去掉静态分析层 / 去掉 LLM 层 / 去掉质量过滤的效果对比）
- [ ] 准备面试 Q&A

---

## 九、面试叙事（最终版）

### 30 秒版本
"我做了一个隐私优先的 PR 测试缺口检测 Agent。聚焦一个问题：PR 改了业务逻辑但没有对应的测试变更。架构上是静态分析前置 + LLM 深度分析的两层设计，工程上是完整的 GitHub App + webhook + async pipeline + trace 系统。模型层面，我在 L20 上 LoRA 微调了 Qwen2.5-Coder-7B 专做这个任务，在我用 bug-fix mining 方法构建的评测集上达到 Claude Sonnet 约 X% 的质量，推理成本是 1/80，支持企业自托管部署。项目已上线，您可以直接安装试用。"

### 30 秒版本里的加分点拆解
1. **垂直聚焦**（"测试缺口检测"而非通用评审）→ 差异化
2. **两层架构**（静态+LLM）→ 系统设计能力
3. **完整工程链**（GitHub App + webhook + async + trace）→ 工程落地能力
4. **LoRA 微调**（Qwen-7B on L20）→ 算法能力 / 双栖定位
5. **Bug-fix Mining Eval**（客观评测方法）→ 科学方法论
6. **成本对比**（1/80 成本达到 X% 质量）→ 工程判断力
7. **已部署可试用** → 产品交付能力

七个独立的加分点，每一个都能深挖 5-10 分钟。

### 高频追问 Q&A

**Q: 为什么只做测试缺口检测，不做通用评审？**
"两个原因。第一，通用评审赛道已经很拥挤了，CodeRabbit、PR-Agent 都在做，我很难做出差异化。测试缺口检测是一个被忽视但高价值的垂直切口——这是 code review 中最常见、最高频导致线上 bug 的问题类型。第二，垂直聚焦让我的 eval 指标可以做到客观量化：缺口检出率、精确率都有清晰定义。通用评审的'好不好'很主观。"

**Q: 静态分析和 LLM 各自负责什么？**
"静态分析处理确定性问题：改了 src/auth.py 但 tests/test_auth.py 没动——这个不需要 LLM，AST 解析就能发现。LLM 处理语义级问题：虽然测试文件改了，但新增的错误处理路径没有对应测试——这个需要理解代码逻辑。两层配合的好处是：40-60% 的缺口由静态分析零成本检出，LLM 只处理需要深度理解的部分，总成本降低。"

**Q: Bug-fix Mining 的 eval 数据怎么保证质量？**
"核心逻辑：如果一个 commit 同时修了 bug 和补了测试，说明之前的代码确实有测试缺口。我用 git blame 追溯到引入 bug 的 PR，把它作为输入，检查 agent 能不能发现这个缺口。Ground truth 是客观的——bug 确实存在，测试确实缺失。每个 case 我还人工验证了 fix commit 新增的测试确实能暴露原始 bug。目前有 50+ 个 case。"

**Q: 微调效果怎么样？**
"在我的 eval 集上，SFT 后的 Qwen-7B 达到 Claude Sonnet 约 X% 的检测率。主要差距在复杂场景——涉及多文件关联的测试缺口，7B 模型理解得不够深。但对于'改了函数没加测试'这类直接的缺口，7B 和 Claude Sonnet 基本持平。考虑到推理成本只有 Claude 的 1/80，这个 trade-off 对企业自托管场景是合理的。"

**Q: 如果微调效果不好怎么办？**
"我跑了三轮实验，详细记录了每轮的参数和结果。第一轮效果确实不好——模型倾向于生成泛泛的评论，不够精确。分析后发现是训练数据中 'no gap found' 的负样本太少，模型学会了'逢改必报'。第二轮调整了正负样本配比（1:1），检测率下降了一些但精确率大幅提升。这个调参过程本身就展示了我对训练数据设计的理解。"

---

## 十、产出清单

| 产出 | 说明 | 重要程度 |
|------|------|----------|
| GitHub 仓库 | Clean code，有 CI，有单元测试（>70%覆盖率） | ⭐⭐⭐⭐⭐ |
| 部署地址 | GitHub App，面试官可安装试用 | ⭐⭐⭐⭐⭐ |
| Eval 报告 | Bug-fix mining 评测集 + 三模型对比数据 | ⭐⭐⭐⭐⭐ |
| SFT 模型 | LoRA adapter + 训练脚本 + 训练日志 | ⭐⭐⭐⭐ |
| HTML 报告页 | 每次评审的 trace + 成本明细 | ⭐⭐⭐⭐ |
| README | 架构图 + 安装指南 + 设计决策 + 评测结果 | ⭐⭐⭐⭐ |
| 技术博客 | 一篇，核心叙事 | ⭐⭐⭐ |
| 演示视频 | 2-3 分钟，给不方便安装的面试官 | ⭐⭐⭐ |
| 消融实验 | 每个模块的贡献度数据 | ⭐⭐⭐ |

---

## 十一、成本估算

| 项目 | 预估成本 |
|------|---------|
| API 调用（开发+调试） | ~$30 |
| API 调用（eval 多轮） | ~$20 |
| 部署（Railway/Fly.io） | $0-5/月（免费额度） |
| 域名 | ~$10/年 |
| SFT 训练（L20 电费） | 忽略不计 |
| **总计** | **~$60-80** |
