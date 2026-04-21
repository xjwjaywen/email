# SWE-bench 代码修复 Agent：完整项目规划

## 一、项目定位

### 项目名称（暂定）
**CodeSurgeon** —— 基于结构化搜索与静态分析增强的代码修复 Agent

### 一句话描述
一个面向 SWE-bench 基准测试的代码修复 Agent，通过结构化故障定位和搜索树引导的修复策略，在不依赖多 Agent 编排的前提下，提升代码修复的准确率和可解释性。

### 为什么不做多 Agent
你的简历上已经有三个多 Agent 项目（Darwinian、NeuroImage Agent、多智能体研究系统），全部基于 LangGraph 编排。这个项目故意采用**单 Agent + 结构化流水线**的范式，展示你对不同架构范式的理解和选择判断力。面试时能说清"什么场景适合多 Agent、什么场景单 Agent 更优"，比再做一个多 Agent 项目价值高得多。

### 核心差异化
不是"又一个 SWE-bench agent"。差异化来自两个技术点：
1. **静态分析增强的故障定位**：不是纯靠 LLM 猜哪个文件有问题，而是用 AST 解析 + 依赖图分析来缩小搜索范围，再让 LLM 在精确的候选集上做判断。
2. **搜索树引导的修复策略**：不是生成一个补丁就完事，而是受 SWE-Search（ICLR 2025）启发，维护一个轻量的搜索树，探索多条修复路径，用测试反馈来剪枝和引导。

---

## 二、关键背景知识

### SWE-bench 是什么
SWE-bench 是 Princeton 团队发布的基准测试，收录了来自 12 个主流 Python 开源项目（Django、Flask、scikit-learn、sympy 等）的真实 GitHub issue。每个 issue 包含：
- issue 描述（问题是什么）
- 对应的代码仓库（某个历史 commit 的快照）
- 测试补丁（用于验证修复是否正确）

Agent 的任务是：读取 issue 描述 → 分析代码库 → 生成修复补丁 → 补丁必须通过测试。

### 评测集选择
- **SWE-bench Verified**：500 个经人工审核的高质量 case，是目前主流排行榜使用的评测集。建议以此为主要评测目标。
- **SWE-bench Lite**：300 个相对简单的 case，可用于开发阶段快速迭代验证。

### 现有方案的核心思路

| 方案 | 核心思路 | 典型得分（Verified） | 特点 |
|------|----------|---------------------|------|
| mini-swe-agent | 100 行代码，只用 bash，靠模型能力 | 74%（Claude Sonnet 4） | 极简，证明模型能力是主要因素 |
| SWE-agent | 精心设计的 Agent-Computer Interface | ~23%（开源模型） | NeurIPS 2024，注重交互界面设计 |
| Agentless | 无 agent 循环，分层定位 + 采样修复 | ~32%（Lite） | 证明不需要复杂 agent 也能做 |
| SWE-Search | MCTS 搜索树 + 多 agent 评估 | 比基线提升 23% | ICLR 2025，搜索策略创新 |

### 关键洞察
mini-swe-agent 的结果说明：**用顶级模型 + 极简架构就能达到很高分数**。这意味着：
- 在使用同等模型的前提下，单纯的架构改进空间有限
- 真正的差异化要来自**在更便宜的模型上，通过更好的策略逼近高分模型的效果**
- 或者来自**可解释性和分析深度**——不只是跑分，而是理解为什么能修、为什么不能修

---

## 三、系统架构设计

### 总体流程

```
Issue 描述
    │
    ▼
┌─────────────────────┐
│  Phase 1: 故障定位   │  ← 静态分析增强
│  (Fault Localization)│
└─────────┬───────────┘
          │ 输出：候选位置列表（文件/函数/行号 + 置信度）
          ▼
┌─────────────────────┐
│  Phase 2: 上下文组装  │  ← Token 预算管理
│  (Context Assembly)  │
└─────────┬───────────┘
          │ 输出：精选的代码上下文
          ▼
┌─────────────────────┐
│  Phase 3: 修复搜索   │  ← 搜索树 + 测试反馈
│  (Repair Search)     │
└─────────┬───────────┘
          │ 输出：候选补丁集合
          ▼
┌─────────────────────┐
│  Phase 4: 补丁验证   │  ← Docker 沙箱
│  (Patch Validation)  │
└─────────┘
          │ 输出：最终补丁
```

### 设计原则
1. **不用 LangGraph / LangChain**：用纯 Python 实现，所有流程控制自己写。展示你不依赖框架也能构建可靠系统。
2. **单 Agent 循环**：一个 LLM 贯穿始终，不做角色切换。通过不同阶段的 prompt 模板来引导行为。
3. **工具最小化**：只提供必要的工具（文件读取、搜索、编辑、bash 执行），不做花哨的工具设计。
4. **可解释优先**：每一步决策都有结构化的日志，方便事后分析。

---

## 四、各阶段详细设计

### Phase 1：故障定位（Fault Localization）

这是整个系统最核心的阶段。Agentless 论文的核心发现是：**精确的故障定位是自动修复的关键**。定位准了，修复通常不难；定位错了，后面全白费。

#### 1.1 Issue 信息提取
```python
def extract_issue_info(issue_text: str) -> IssueInfo:
    """
    从 issue 描述中提取结构化信息：
    - error_messages: 报错信息、stack trace
    - mentioned_files: 明确提到的文件路径
    - mentioned_symbols: 提到的函数名、类名、变量名
    - expected_behavior: 期望行为描述
    - actual_behavior: 实际行为描述
    - reproduction_steps: 复现步骤（如果有）
    """
```
这一步用 LLM 做结构化提取，prompt 要求输出 JSON 格式。

#### 1.2 仓库结构索引（静态分析核心）
```python
def build_repo_index(repo_path: str) -> RepoIndex:
    """
    用 Python ast 模块解析整个仓库，构建索引：
    
    1. 文件级索引：
       - 文件路径 → 文件描述（首行 docstring / 模块注释）
       - 文件路径 → 包含的类和函数列表
    
    2. 符号级索引：
       - 符号名 → 定义位置（文件:行号）
       - 符号名 → 类型（class / function / method）
       - 符号名 → 签名（参数列表 + 返回类型注解）
    
    3. 依赖图：
       - 文件 A import 了文件 B 的哪些符号
       - 函数 A 调用了哪些函数
       - 类 A 继承了哪些类
    """
```

关键实现细节：
- 使用 Python 标准库 `ast` 模块，不需要额外依赖
- 对大型仓库（如 Django），只解析 Python 文件，跳过测试目录、文档目录
- 索引结果序列化存储，避免重复解析
- 依赖图用邻接表表示，支持快速查询"谁调用了这个函数"和"这个函数调用了谁"

#### 1.3 分层定位
受 Agentless 启发，采用从粗到细的三层定位：

**第一层：文件定位**
```
输入：issue 信息 + 仓库文件列表（只含路径和简短描述）
输出：Top-5 可疑文件，附理由
```
- 先用 issue 中提到的文件名、符号名做精确匹配（在索引中查询）
- 将匹配结果 + 仓库文件树摘要一起送给 LLM，让它判断最可能出问题的文件
- 如果 issue 包含 stack trace，直接提取其中的文件路径作为高优先级候选

**第二层：函数/类定位**
```
输入：Top-5 文件的类和函数列表（从索引中获取，含签名和 docstring）
输出：Top-5 可疑函数/类，附理由
```
- 不需要把整个文件内容送给 LLM，只发送函数签名和 docstring
- 这极大地节省了 token，允许在更大范围内搜索

**第三层：行级定位**
```
输入：Top-5 函数/类的完整源码 + 依赖图中的直接关联代码
输出：具体的可疑代码行范围 + 初步修复方向判断
```
- 此时才把完整代码发送给 LLM
- 同时发送依赖图中的上下游代码（调用者/被调用者的关键片段）

#### 1.4 静态分析增强点

以下是纯 LLM 定位做不好、但静态分析能帮上忙的地方：

**Import 链追踪：** Issue 报的是 A 模块的错误，但 root cause 在 A 导入的 B 模块里。纯 LLM 可能只关注 A 文件，但依赖图可以自动把 B 纳入候选。

**继承链分析：** 子类的 bug 可能源于父类方法的修改。AST 解析可以自动找到继承关系，把父类/子类都纳入搜索范围。

**符号交叉引用：** Issue 提到 `process_data` 函数行为异常，索引可以快速找到所有定义了 `process_data` 的位置（可能有多个同名函数在不同模块中）。

### Phase 2：上下文组装（Context Assembly）

#### 2.1 Token 预算管理
```python
class TokenBudgetManager:
    """
    将总 token 预算分配给不同部分：
    
    总预算（假设使用 128K 上下文模型）：
    - system prompt: ~2K tokens（固定）
    - issue 描述: ~1K tokens（固定）
    - 代码上下文: ~30K tokens（动态分配）
    - 修复历史: ~5K tokens（随迭代增长）
    - 预留响应: ~4K tokens（固定）
    
    代码上下文的分配策略：
    - 主要可疑文件: 60% 预算
    - 关联文件（import/继承相关）: 25% 预算
    - 测试文件（理解期望行为）: 15% 预算
    """
```

#### 2.2 智能上下文裁剪
不是简单地截断文件，而是：
- 对于大文件，只保留可疑函数 + 前后各 20 行 + 文件级 import 语句
- 对于关联文件，只保留被引用的符号定义
- 对于测试文件，只保留和 issue 相关的测试方法（通过测试方法名和 issue 关键词匹配）

```python
def assemble_context(
    issue_info: IssueInfo,
    localization_result: LocalizationResult,
    repo_index: RepoIndex,
    token_budget: int
) -> str:
    """
    组装发送给 LLM 的代码上下文。
    按优先级从高到低填充 token 预算：
    1. 可疑位置的完整代码
    2. 直接依赖的代码片段
    3. 相关测试代码
    4. 项目级上下文（README 摘要、代码规范等）
    """
```

### Phase 3：修复搜索（Repair Search）

这是项目的第二个核心技术亮点。不做单次生成，而是维护一个搜索树。

#### 3.1 搜索树结构
```python
@dataclass
class RepairNode:
    """搜索树的节点"""
    patch: str                    # 生成的补丁（diff 格式）
    reasoning: str                # LLM 的修复推理过程
    test_result: TestResult       # 测试执行结果
    parent: Optional['RepairNode']  # 父节点
    children: list['RepairNode']    # 子节点（基于此补丁的改进尝试）
    value_score: float            # 评估分数
    
class RepairTree:
    """修复搜索树"""
    root: RepairNode
    max_depth: int = 3            # 最大搜索深度
    max_children: int = 3         # 每个节点最多展开的子节点数
    total_budget: int = 10        # 总共最多尝试的补丁数
```

#### 3.2 搜索流程
```
Round 1: 基于定位结果，生成 3 个不同方向的候选补丁
         ├── Patch A: 修改方式 1（如：修复条件判断）
         ├── Patch B: 修改方式 2（如：添加参数校验）
         └── Patch C: 修改方式 3（如：修改返回值处理）
         
         对每个补丁运行测试，收集结果

Round 2: 选择最有希望的补丁（部分测试通过/错误信息最有信息量的），
         基于测试反馈生成 2-3 个改进版本
         ├── Patch A1: 基于 A 的测试错误信息修正
         ├── Patch A2: 基于 A 的另一种修正方向
         └── Patch B1: 基于 B 的测试错误信息修正

Round 3: 继续深化，或者回退到新的定位结果重新尝试
```

#### 3.3 搜索策略的关键设计

**多样性保障：** 每轮生成的候选补丁必须涉及不同的修复策略。在 prompt 中明确要求"给出三种不同的修复方向"，并提供之前已尝试的方向列表，避免重复。

**测试反馈利用：** 这是比盲目重试更有价值的地方。测试失败信息包含了丰富的诊断线索：
```python
def analyze_test_feedback(test_result: TestResult) -> FeedbackAnalysis:
    """
    分析测试反馈，提取有用信息：
    - 哪些测试通过了（修复可能部分正确）
    - 失败测试的错误类型（AssertionError / TypeError / ImportError 等）
    - 错误信息中的关键线索
    - 与原始测试结果相比，是变好了还是变差了
    """
```

**预算控制：** 总共最多生成 10 个补丁（3+3+3+1 的分配），严格控制 API 调用成本。每个补丁的生成都记录 token 消耗和耗时。

**剪枝规则：**
- 如果某个补丁导致更多测试失败（比原始代码还差），不再展开它
- 如果一个修复方向的两个变体都失败且错误信息相似，放弃这个方向
- 如果已经有补丁通过了所有相关测试，提前终止搜索

### Phase 4：补丁验证（Patch Validation）

#### 4.1 Docker 沙箱执行
```python
class SandboxExecutor:
    """
    在 Docker 容器中执行测试。
    
    每个 issue 对应一个独立容器：
    - 基于 SWE-bench 提供的环境镜像
    - 预装项目依赖
    - 应用补丁 → 运行测试 → 收集结果
    - 超时限制：300 秒
    - 资源限制：1 CPU / 2GB 内存
    """
```

#### 4.2 结果判定
```python
def validate_patch(patch: str, instance: SWEBenchInstance) -> ValidationResult:
    """
    三层验证：
    1. 语法检查：补丁能否干净地 apply（git apply --check）
    2. 目标测试：issue 对应的测试用例是否通过
    3. 回归测试：原本通过的测试是否仍然通过（不引入新 bug）
    """
```

---

## 五、技术栈

| 组件 | 技术选型 | 理由 |
|------|----------|------|
| 语言 | Python 3.10+ | SWE-bench 生态全是 Python |
| LLM 调用 | 直接用 httpx 调 API | 不引入 LangChain 等框架，保持轻量 |
| 模型 | DeepSeek-V3 / Claude Sonnet 作对比 | 便宜模型 vs 贵模型，对比展示架构价值 |
| AST 解析 | Python ast 标准库 | 零依赖，够用 |
| Token 计数 | tiktoken | 你在实习中用过，精确计数 |
| 沙箱 | Docker + SWE-bench harness | 官方评测方式 |
| 日志/追踪 | 自建结构化日志（JSON 格式） | 轻量，无外部依赖 |
| 评测 | SWE-bench 官方评测脚本 | 结果可与排行榜直接对比 |

**不用的东西（以及为什么不用）：**
- LangGraph / LangChain：故意不用，展示裸写 agent 的能力
- FAISS / 向量检索：代码搜索用 AST 索引 + 关键词匹配更精确，不需要语义搜索
- Langfuse：用自建日志系统，更轻量，按需设计

---

## 六、项目结构

```
codesurgeon/
├── README.md
├── pyproject.toml
│
├── codesurgeon/
│   ├── __init__.py
│   ├── main.py                  # 入口：接收 issue，输出补丁
│   │
│   ├── localization/            # Phase 1: 故障定位
│   │   ├── issue_parser.py      # Issue 信息结构化提取
│   │   ├── repo_indexer.py      # AST 解析 + 仓库索引构建
│   │   ├── dependency_graph.py  # Import/调用/继承依赖图
│   │   ├── localizer.py         # 三层分层定位逻辑
│   │   └── static_analyzer.py   # 符号交叉引用、继承链分析
│   │
│   ├── context/                 # Phase 2: 上下文组装
│   │   ├── budget_manager.py    # Token 预算分配
│   │   ├── code_extractor.py    # 智能代码片段提取
│   │   └── assembler.py         # 上下文组装
│   │
│   ├── repair/                  # Phase 3: 修复搜索
│   │   ├── search_tree.py       # 搜索树数据结构
│   │   ├── patch_generator.py   # 补丁生成（调用 LLM）
│   │   ├── feedback_analyzer.py # 测试反馈分析
│   │   └── search_strategy.py   # 搜索策略（展开/剪枝/选择）
│   │
│   ├── execution/               # Phase 4: 补丁验证
│   │   ├── sandbox.py           # Docker 沙箱管理
│   │   ├── test_runner.py       # 测试执行和结果收集
│   │   └── validator.py         # 三层验证逻辑
│   │
│   ├── llm/                     # LLM 调用封装
│   │   ├── client.py            # API 调用（支持 DeepSeek / Claude / GPT）
│   │   ├── prompts.py           # 所有 prompt 模板集中管理
│   │   └── parser.py            # LLM 输出解析（JSON 提取 + fallback）
│   │
│   └── utils/
│       ├── logger.py            # 结构化日志
│       ├── token_counter.py     # Token 计数
│       └── diff_utils.py        # Diff 生成和应用
│
├── evaluation/
│   ├── run_benchmark.py         # 批量运行 SWE-bench 评测
│   ├── analyze_results.py       # 结果分析和统计
│   └── ablation.py              # 消融实验脚本
│
├── prompts/                     # Prompt 模板（Jinja2 格式）
│   ├── localize_files.j2
│   ├── localize_functions.j2
│   ├── localize_lines.j2
│   ├── generate_patch.j2
│   └── refine_patch.j2
│
└── tests/                       # 单元测试
    ├── test_repo_indexer.py
    ├── test_dependency_graph.py
    ├── test_search_tree.py
    └── test_budget_manager.py
```

---

## 七、Prompt 设计要点

### 故障定位 Prompt 示例（文件级）
```
你是一个经验丰富的软件工程师，正在诊断一个开源项目的 bug。

## Issue 描述
{issue_description}

## 从 Issue 中提取的关键信息
- 错误信息: {error_messages}
- 提到的符号: {mentioned_symbols}
- 期望行为: {expected_behavior}

## 项目文件结构（仅显示 Python 源文件）
{file_tree}

## 符号匹配结果
以下文件包含 issue 中提到的符号:
{symbol_matches}

## 任务
请判断最可能包含 bug 的 5 个文件。对每个文件给出：
1. 文件路径
2. 可疑程度（high / medium / low）
3. 判断理由（一句话）

以 JSON 格式输出。
```

### 修复生成 Prompt 示例
```
你是一个经验丰富的软件工程师，正在修复一个 bug。

## Issue 描述
{issue_description}

## 可疑代码位置
文件: {file_path}
函数: {function_name}
```python
{suspicious_code}
```

## 相关代码上下文
{related_code_context}

## 之前尝试过的修复（避免重复）
{previous_attempts_summary}

## 任务
请生成一个修复补丁。要求：
1. 以 unified diff 格式输出
2. 最小化修改范围，只改必要的地方
3. 解释你的修复思路

注意：生成一个与之前尝试不同的修复方向。
```

---

## 八、评测方案

### 8.1 主要指标
- **Resolve Rate（解决率）**：在 SWE-bench Verified 上通过测试的 issue 比例。这是核心指标。
- **成本**：每个 issue 的平均 API 调用成本（美元）
- **耗时**：每个 issue 的平均处理时间

### 8.2 消融实验（Ablation Study）
这是展示技术深度的关键，证明每个设计决策都有贡献：

| 实验 | 说明 |
|------|------|
| Full System | 完整系统 |
| - Static Analysis | 去掉 AST 索引和依赖图，纯 LLM 做文件定位 |
| - Search Tree | 去掉搜索树，只生成一个补丁 |
| - Test Feedback | 去掉测试反馈驱动的迭代修复 |
| - Smart Context | 去掉智能上下文裁剪，改为简单截断 |

每组实验都在相同的 case 子集上运行，对比 Resolve Rate 的变化。

### 8.3 跨模型对比
同一套架构，分别用：
- DeepSeek-V3（便宜）
- Claude Sonnet（中等）
- Claude Opus / GPT-4o（贵）

如果你的架构在便宜模型上的提升幅度大于在贵模型上的提升幅度，说明**架构设计对弱模型的增益更大**——这是一个很有说服力的结论。

### 8.4 失败案例分析
对未能解决的 case 做分类分析：
- 定位失败（没找到正确的文件/函数）占比多少？
- 定位正确但修复错误占比多少？
- 修复正确但引入了回归 bug 占比多少？
- 需要跨文件修改（当前架构不擅长）占比多少？

这个分析比最终分数更有价值，展示你对系统瓶颈的深刻理解。

---

## 九、实现路线图

### 第 1-2 周：基础设施 + 故障定位

**Week 1：环境搭建 + 仓库索引**
- [ ] 搭建 SWE-bench 评测环境（Docker 镜像拉取、评测脚本配置）
- [ ] 手动跑通 3-5 个 case 的完整评测流程，理解数据格式
- [ ] 实现 repo_indexer.py：AST 解析，生成文件级和符号级索引
- [ ] 实现 dependency_graph.py：import 关系图、函数调用图

**Week 2：分层定位**
- [ ] 实现 issue_parser.py：从 issue 描述中提取结构化信息
- [ ] 实现三层定位逻辑（文件 → 函数 → 行级）
- [ ] 在 20 个 case 上手动验证定位准确率
- [ ] 调优定位相关的 prompt

**Week 2 检查点：** 在 20 个 case 上，Top-5 文件定位准确率应 ≥ 70%（即正确文件在 Top-5 候选中）

### 第 3-4 周：上下文组装 + 基本修复

**Week 3：上下文管理 + 单次修复**
- [ ] 实现 Token 预算管理器
- [ ] 实现智能代码片段提取和上下文组装
- [ ] 实现基本的补丁生成（单次，不迭代）
- [ ] 实现 Docker 沙箱测试执行

**Week 4：端到端打通**
- [ ] 串联所有模块，实现完整的单次修复流水线
- [ ] 在 SWE-bench Lite 的一个小子集（50 个 case）上跑通评测
- [ ] 记录基线分数

**Week 4 检查点：** 端到端流水线能跑通，在 50 个 case 上拿到一个基线分数（即使很低也没关系）

### 第 5-6 周：搜索树 + 迭代修复

**Week 5：搜索树实现**
- [ ] 实现搜索树数据结构
- [ ] 实现测试反馈分析模块
- [ ] 实现多候选补丁生成（每轮 3 个方向）
- [ ] 实现基于反馈的补丁改进逻辑

**Week 6：搜索策略调优**
- [ ] 实现剪枝规则
- [ ] 调优搜索深度和宽度参数
- [ ] 在更大的 case 集上运行，对比搜索树 vs 单次修复的效果
- [ ] 优化 token 消耗（搜索树会显著增加 API 调用）

**Week 6 检查点：** 搜索树方案的 Resolve Rate 应比单次修复基线有可见提升

### 第 7-8 周：评测 + 分析 + 打磨

**Week 7：完整评测**
- [ ] 在 SWE-bench Verified（500 case）上运行完整评测
- [ ] 跑消融实验（至少 4 组对比）
- [ ] 跑跨模型对比实验
- [ ] 统计成本和耗时

**Week 8：分析 + 文档**
- [ ] 失败案例分类分析
- [ ] 整理实验数据，制作对比图表
- [ ] 写 README（项目动机、架构设计、实验结果、关键发现）
- [ ] 准备面试讲解的叙事线

---

## 十、面试叙事线

### 30 秒版本
"我做了一个 SWE-bench 代码修复 Agent。和现有方案的区别是：我用 AST 静态分析来增强故障定位的准确率，用搜索树来探索多条修复路径，让测试反馈来引导搜索方向。在 SWE-bench Verified 上用 DeepSeek 跑出了 X% 的解决率。"

### 面试官可能追问的问题 + 你的回答方向

**Q: 你和 SWE-agent 有什么区别？**
A: SWE-agent 的核心创新是 Agent-Computer Interface 的设计——给 agent 提供更好用的工具。我的创新点在另一个维度：故障定位策略和修复搜索策略。SWE-agent 用的是线性探索（agent 一步步决定下一步做什么），我用的是树形搜索（同时探索多条路径，用测试反馈来剪枝）。

**Q: 为什么不用多 Agent？**
A: 我之前做过三个多 Agent 项目（Darwinian、NeuroImage Agent 等），发现多 Agent 引入了额外的通信开销和上下文切换成本。对于代码修复这个任务，瓶颈在于定位和搜索策略，不在于角色分工。单 Agent + 好的搜索策略比多个 Agent 互相传话更高效。我的消融实验也验证了这一点。

**Q: 静态分析具体帮了多少？**
A: 在我的消融实验中，去掉静态分析模块后，文件级定位的 Top-5 准确率从 X% 下降到 Y%，最终 Resolve Rate 下降了 Z 个百分点。主要增益来自两个场景：一是 issue 描述比较模糊、没有明确指出文件路径的情况；二是 bug 的 root cause 在被调用方而不是调用方的情况——依赖图可以自动追踪到这类间接关联。

**Q: 搜索树的开销大吗？**
A: 平均每个 issue 生成 6-8 个候选补丁（对比单次修复的 1 个），API 调用成本大约是 3-4 倍。但 Resolve Rate 的提升超过了成本的增加，整体"每个正确修复的成本"反而下降了。我做了搜索预算的参数实验，发现预算从 5 增加到 10 有明显提升，但从 10 增加到 15 收益递减。

**Q: 你的分数和排行榜上的比怎么样？**
A: 排行榜顶部是用最贵的闭源模型跑的（Claude Opus 4），我的系统侧重点不同——在模型成本约束下最大化修复能力。用 DeepSeek 跑出 X%，换成 Claude Sonnet 可以到 Y%。我更关注的是架构层面每个组件的贡献度，消融实验证明了静态分析和搜索树分别贡献了多少增益。

---

## 十一、成本估算

### API 调用成本
假设使用 DeepSeek-V3（输入 $0.27/M tokens，输出 $1.10/M tokens）：
- 每个 issue 平均 4 次 LLM 调用（定位） + 8 次（修复搜索） ≈ 12 次调用
- 每次平均输入 8K tokens，输出 2K tokens
- 每个 issue 成本 ≈ $0.05
- 500 个 case（Verified 全量）≈ $25
- 加上消融实验和调试，总预算控制在 $100-150

### 硬件需求
- Docker 环境：建议至少 16GB 内存、50GB 可用磁盘（SWE-bench 镜像较大）
- 不需要 GPU
- 可以在个人电脑或云服务器上运行

---

## 十二、风险和应对

| 风险 | 应对 |
|------|------|
| AST 解析在某些项目上失败（语法不标准） | 对解析失败的文件退化为纯文本处理，不阻塞流程 |
| 搜索树没有带来显著提升 | 如果数据证明单次修复就够了，这本身也是一个有价值的发现。在论文/面试中诚实报告 |
| DeepSeek 分数太低不好看 | 同时报告 Claude Sonnet 的分数，用跨模型对比展示架构价值 |
| Docker 环境配置耗时长 | 第一周全力搞定环境，用 SWE-bench 官方文档和 swebench-docker 工具 |
| Token 消耗超预算 | 设置严格的 per-issue token 上限，超限自动终止 |
