# RealAgent 自主 Multi-Agent 使用指南

## 设计目标

RealAgent 提供自然语言入口，由 LLM 自行判断是否以及按什么顺序调用专业 Agent。它与原有 Hunter、Miner、Validator、Coach 模块及固定工作流并存，不影响原有 API。

## 架构

`AutonomousResearchGraph` 使用 LangGraph `create_react_agent` 构建监督者。监督者可调用四个 LangChain `StructuredTool`：`delegate_to_hunter`（检索论文）、`delegate_to_miner`（分析论文）、`delegate_to_validator`（引用校验）、`delegate_to_coach`（写作）。每个工具都委派给已有模块化 Agent。监督者阅读工具结果后决定下一步、重试、改换 Agent 或结束，而不是执行固定流水线。

LangGraph 内存 checkpointer 以 `thread_id` 隔离上下文；ReAct 递归上限为 30，避免无限调用。

## 启动与配置

按照项目原有方式配置 OpenAI 兼容模型（API Key、Base URL、模型名），然后运行：

```bash
python run.py
```

打开 `http://localhost:8000`，点击“RealAgent 自主研究助理”，输入自然语言任务。原有模块卡片仍可单独使用。

示例：“搜索 3 篇关于 LangGraph multi-agent 的近期论文，比较创新点，并生成 BibTeX 引用。”

## API

提交异步任务：

```http
POST /api/v1/agent/runs
Content-Type: application/json

{"message":"解释 RAG 和 agentic RAG 的区别，并润色为论文引言","thread_id":"optional-id","context":{}}
```

返回 HTTP 202 和 run id。使用 `GET /api/v1/agent/runs/{run_id}` 查询进度，或用 `GET /api/v1/agent/runs?limit=20` 查看最近任务。`status` 为 `queued`、`running`、`completed` 或 `failed`；`events` 是按时间排序的处理轨迹；`answer` 是最终回答。

## 日志与工作台

后端关键事件使用 `[real-agent:<run_id>]` 前缀记录，包括监督者规划、专业 Agent 开始、完成和失败。前端工作台展示同一组结构化事件。

当前 run/event 存储在进程内存，适合单实例开发。生产环境建议持久化到项目已有 Redis/PostgreSQL，并为接口增加用户鉴权。

## 已验证环境

2026-07-11 在本地 Conda `agent` 环境完成以下验证：

- LangChain、LangGraph、FastAPI 和四个专业 Agent 运行时导入；
- RealAgent 三个 API 路由注册；
- 全项目 Python bytecode 编译与 `git diff --check`；
- 使用项目 `.env` 中的 OpenAI 兼容端点执行真实 LangGraph 监督者请求，模型按要求返回 `E2E_OK`，并产生 `planning`、`completed` 进度事件。

该冒烟测试刻意不调用外部检索工具，用于验证“配置加载 → LLM → LangGraph → 最终消息 → 事件轨迹”的主链路。论文检索、PDF 下载等外部服务仍受各服务网络和凭据状态影响。

## 检索来源与失败降级

RealAgent 的 Hunter 默认只使用无需密钥的 ArXiv。只有用户明确要求 IEEE 时，监督者才应传入 `sources: ["ieee"]` 或 `sources: ["arxiv", "ieee"]`。IEEE 需要在 `.env` 配置 `IEEE_API_KEY`。

各检索来源相互隔离：某个来源失败或缺少凭据时，Hunter 会继续执行其他来源，并在结果的 `source_errors` 中说明跳过原因；已有的 ArXiv 结果不会因 IEEE 失败而丢失。`source_results` 展示各成功来源取得的条目数，`partial_success` 表示任务获得了部分结果但至少一个来源失败。

前端进度区以时间线展示监督者规划、专业 Agent 调用、完成与失败；最终回答单独显示，不再把内部 trace 作为原始 JSON 混入答案。

### 开发进度与超时

RealAgent 事件流会展示 `decision`（LLM 选择的 Agent、函数和任务）、`agent_progress`（专业 Agent 内部函数与阶段）、`agent_completed/agent_failed`（含耗时）以及 `duplicate_skipped`（相同参数的重复调用被跳过）。后端日志同步输出结构化 `data`，长摘要和正文只记录长度，不打印全文。

监督者调用专业 Agent 时统一应用 `AGENT_TIMEOUT` 对应的 `agent_timeout`（当前默认 300 秒）。Hunter 会报告来源检索、去重筛选以及逐篇 PDF 处理进度。完全相同的 Agent 输入在同一次请求内只执行一次，避免模型重复消耗外部 API 和下载时间。

## 论文标识约定

跨 Agent 传递论文时必须区分数据库标识与来源标识：`db_id` 是 PostgreSQL UUID；`external_id` 是 `2606.01899v1` 这样的 ArXiv ID 或 IEEE 平台 ID；`source` 表示来源。Hunter 会为每篇论文返回可直接交给 Miner 的 `analysis_input`。

Miner 会先校验数据库 ID。非 UUID 的旧版 `paper_id` 不再进入 PostgreSQL，而会按 ArXiv 外部 ID 解析；分析报告和向量索引也只会使用合法 UUID 写入数据库。新代码应优先原样传递 `analysis_input`，不要把 `external_id` 改名为 `db_id`。

## 真实证据门控

检索来源名称不区分大小写，`ArXiv`、`arxiv` 均会实际调用 ArXiv API。Hunter 返回的论文元数据来自外部 API；即使 PDF 下载失败，标题、作者、摘要、来源和外部 ID 仍作为检索证据返回，并单独记录下载数量。

监督者只允许使用 Hunter 返回的论文事实。如果本次调用过 Hunter 但没有取得任何带标题和外部 ID 的真实论文，最终的模型文本会被确定性证据门控替换为“未获得可验证的论文结果”，同时产生 `evidence_blocked` 事件。系统不会再使用模型记忆虚构论文标题、摘要、方法或创新点。成功检索会产生 `evidence` 事件及论文数量。

RealAgent 最终回答使用前端内置的 Markdown 渲染器展示，支持标题、列表、粗体、斜体、行内代码和链接，不再显示未经渲染的 Markdown 原文。

最终回答渲染前会兼容 OpenAI 兼容端点可能返回的字面量 `\\n`，并移除包裹整段回答的 ` ```markdown ` 代码围栏。入口 `index.html` 使用 `Cache-Control: no-store`，避免浏览器继续执行旧版渲染逻辑。可在开发者工具中检查最终回答元素的 `data-markdown-renderer="realagent-v2"` 确认新版本已生效。

## 扩展专业 Agent

在 `agents/autonomous.py` 的工具列表注册 specialist tool，并在 `AgentController.agents` 提供实现。工具描述应明确适用场景和必需字段，专业 Agent 的 `run(input_data)` 应返回可 JSON 序列化结果并校验输入。
