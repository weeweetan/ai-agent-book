# Kimi / DeepSeek / OpenAI 配置

先在仓库根目录安装共享包和对应章节依赖（Python 3.11–3.13）：

```bash
python -m pip install -e '.[ch1]'
```

共享配置和请求适配器位于 `agentbook/providers/`。源码中的示例会按脚本位置定位仓库根目录，也支持进入实验目录后用同一个 Python 安装并运行：

```bash
cd chapter1/web-search-agent
python -m pip install -r requirements.txt
python main.py --provider openai '搜索 Python 官方文档'
```

依赖必须安装到运行脚本的 Python 环境中；遇到 `No module named 'dotenv'` 时，对应的发行包是 `python-dotenv`，已在各联网示例的依赖文件中声明。使用 `python -m pip` 可以确保安装器与当前解释器一致。

单项目依赖中的 OpenAI SDK 最低版本与根目录保持一致，均为 `openai>=1.68.0`。CI 同时检查最低版本和当前版本，避免旧 SDK 已满足旧依赖约束，却无法接受共享适配器生成的参数。

## 选择供应商

选一组环境变量即可，不需要同时配置三家的密钥。可写入示例所读取的 `.env`，也可直接导出：

| 供应商 | LLM_PROVIDER | 密钥 | 默认通用模型 | 自定义端点 |
| --- | --- | --- | --- | --- |
| Kimi | `kimi`（也接受 `moonshot`） | `MOONSHOT_API_KEY` / `KIMI_API_KEY` | `kimi-k3` | `KIMI_BASE_URL` |
| DeepSeek | `deepseek` | `DEEPSEEK_API_KEY` | `deepseek-flash` | `DEEPSEEK_BASE_URL` |
| OpenAI | `openai` | `OPENAI_API_KEY` | `gpt-4o` | `OPENAI_BASE_URL` |

```bash
# DeepSeek
export LLM_PROVIDER=deepseek
export DEEPSEEK_API_KEY=your-deepseek-key
python chapter1/learning-from-experience/quick_demo.py

# OpenAI
export LLM_PROVIDER=openai
export OPENAI_API_KEY=your-openai-key
python chapter1/learning-from-experience/quick_demo.py
```

模型优先级为：显式函数参数/CLI `--model` → `LLM_MODEL` → 兼容旧配置的 `MODEL_NAME` → 对应供应商的 `DEEPSEEK_MODEL` / `OPENAI_MODEL` / `KIMI_MODEL` → 示例或供应商默认值。切换供应商时请删除旧 `.env` 中固定为 `kimi-*` 的模型覆盖。部分示例保留自己的 `MODEL`、`TEXT_MODEL`、`VISION_MODEL`、`CHAT_MODEL` 参数；显式模型应与所选端点匹配。

供应商参数优先于 `LLM_PROVIDER`。未指定时保留各示例原来的供应商默认值；不会根据另一家服务的密钥猜测端点。已有 OpenRouter 兜底保留；直连所选供应商密钥存在时，使用其自身端点。

## 接入的通用示例

下列示例的聊天/工具调用路径支持 DeepSeek 和 OpenAI。除标明的入口外，使用 `LLM_PROVIDER` 选择；有独立 `--provider` 的入口可显式传入同名供应商。

| 章节 | 示例目录 |
| --- | --- |
| 1 | `context`、`web-search-agent`、`learning-from-experience` |
| 2 | `context-compression`、`kv-cache`、`system-hint`、`prompt-injection`、`agent-skills-ppt` |
| 3 | `agentic-rag`、`contextual-retrieval`、`agentic-rag-for-user-memory`、`contextual-retrieval-for-user-memory`、`user-memory`、`mem0`、`memobase` |
| 4 | `execution-tools` |
| 5 | `code-for-math`、`code-for-logic`、`dynamic-form`、`erp-agent`、`conversational-ui`、`paper-to-ppt`、`paper-to-video`（讲解文本）、`agent-creator` |
| 6 | `agent-with-event-trigger`、`async-agent` |
| 9 | `prompt-auto-optimization`、`self-evolving-tools` |
| 10 | `parallel-web-research`、`voice-werewolf`（玩家文本）、`autonomous-phone-registration`（决策文本） |

`agent-creator` 也接受 `AGENT_CREATOR_PROVIDER=deepseek`；该变量优先于 `LLM_PROVIDER`。生成出的独立 Agent 仍使用生成器传入的 OpenAI 兼容端点配置。

## 搜索示例

Kimi 路径保留 Moonshot Formula。DeepSeek/OpenAI 的搜索通过相同的 ReAct 循环调用独立 Serper 工具：

```bash
export LLM_PROVIDER=deepseek
export DEEPSEEK_API_KEY=your-deepseek-key
export SERPER_API_KEY=your-serper-key
python chapter1/web-search-agent/main.py '搜索 Python 官方文档，解释 asyncio.TaskGroup'

# CLI 也可以显式选择供应商
python chapter1/web-search-agent/main.py --provider openai '搜索 Python 官方文档'
```

没有 `SERPER_API_KEY` 时，DeepSeek/OpenAI 搜索入口会明确报错，不会把模型记忆当成搜索结果。离线轨迹只使用 Python 标准库，可以先运行，再决定是否安装联网依赖：

```bash
python chapter1/web-search-agent/main.py --provider offline-demo --output demo.json
```

`--help` 同样不加载模型 SDK 或 `.env` 依赖。联网入口缺少依赖时，会给出使用当前 Python 安装 `requirements.txt` 的命令。

## 参数及能力边界

共享 `chat_completion` 适配器保留 SDK 的同步、异步和流式返回类型。OpenAI 推理模型使用 `max_completion_tokens` 并省略不支持的采样参数，详见 [OpenAI Chat Completions 参数](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)。

通用示例默认给 DeepSeek Flash 显式设置 `thinking: disabled`，因为不少教学循环只保存文本和工具调用。需要思考模式时可传 `extra_body={"thinking": {"type": "enabled"}}`，同时必须在工具调用的后续请求中完整保留 `reasoning_content`；共享适配器不会覆盖显式设置。参见 [DeepSeek 思考模式](https://api-docs.deepseek.com/guides/thinking_mode/)。模型名称及视觉支持参见 [DeepSeek 模型说明](https://api-docs.deepseek.com/quick_start/pricing/)。

`chapter1/context` 为比较上下文中的推理信息，会显式开启 DeepSeek 思考模式并保存推理字段；其 `no_reasoning` 模式改为关闭思考，避免删除协议必需字段。该配置与其他模型的消融条件不同，实验指标应分别解释。

聊天接口兼容不代表图像生成、TTS、ASR、embedding、Realtime 或厂商托管工具可以互换。这些路径继续使用各示例配置的专用服务。视觉审核必须选支持图片的模型（例如 DeepSeek Flash，不能用仅文本模型替代）。`mem0` 的 embedding 配置独立于聊天供应商；视频和狼人杀的语音服务也独立配置。

固定模型的论文复现实验、历史验证产物、训练脚本、第三方内置项目保留原来的配置和结果。换模型会影响实验指标，新结果需要重新运行，不能沿用已有报告。

手机注册示例的历史验收记录还绑定 `source_sha256`：`decision.py`、`orchestration.py` 等源码修改后，用当前源码验证旧记录会因哈希不匹配而失败。这是记录来源的完整性检查，不应通过改写历史哈希来消除。要复核旧记录，请向 `validate_acceptance.py --source-root` 提供与记录哈希一致的原源码；要证明新版通过真实服务验收，则需要重新运行实验并保留新的回执。

## 离线验证

```bash
python -m pip install -e '.[dev]'
python -m pytest -q tests/test_example_providers.py tests/test_providers.py
cd chapter1/web-search-agent
python -m pytest -q
```

测试使用假密钥和模拟 HTTP 传输，检查端点、认证头、模型选择、工具往返及同步/异步流式协议，不调用付费 API。真实账户的模型权限和网络连通性需要使用自己的密钥另行验证。

`checkout-startup tests` 工作流单独安装示例入口所需的轻量依赖，刻意不安装 `agentbook`，在 31 个实验目录分别启动配置模块并检查 Kimi、DeepSeek、OpenAI 三组配置。搜索示例还通过 `python -E -S main.py --provider offline-demo` 验证没有第三方包时的离线路径。这些启动检查不替代依赖外部数据库、媒体服务或真实模型的完整实验验收。
