# Kimi / DeepSeek / OpenAI 配置

先在仓库根目录安装共享包和对应章节依赖（Python 3.11–3.13）：

```bash
python -m pip install -e '.[ch1]'
```

如果只安装了某个实验的 `requirements.txt`，还需要在根目录执行 `python -m pip install -e .`。共享配置和请求适配器位于 `agentbook/providers/`。

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

没有 `SERPER_API_KEY` 时，DeepSeek/OpenAI 搜索入口会明确报错，不会把模型记忆当成搜索结果。离线轨迹仍可用 `--provider offline-demo` 查看。

## 参数及能力边界

共享 `chat_completion` 适配器保留 SDK 的同步、异步和流式返回类型。OpenAI 推理模型使用 `max_completion_tokens` 并省略不支持的采样参数，详见 [OpenAI Chat Completions 参数](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)。

通用示例默认给 DeepSeek Flash 显式设置 `thinking: disabled`，因为不少教学循环只保存文本和工具调用。需要思考模式时可传 `extra_body={"thinking": {"type": "enabled"}}`，同时必须在工具调用的后续请求中完整保留 `reasoning_content`；共享适配器不会覆盖显式设置。参见 [DeepSeek 思考模式](https://api-docs.deepseek.com/guides/thinking_mode/)。模型名称及视觉支持参见 [DeepSeek 模型说明](https://api-docs.deepseek.com/quick_start/pricing/)。

`chapter1/context` 为比较上下文中的推理信息，会显式开启 DeepSeek 思考模式并保存推理字段；其 `no_reasoning` 模式改为关闭思考，避免删除协议必需字段。该配置与其他模型的消融条件不同，实验指标应分别解释。

聊天接口兼容不代表图像生成、TTS、ASR、embedding、Realtime 或厂商托管工具可以互换。这些路径继续使用各示例配置的专用服务。视觉审核必须选支持图片的模型（例如 DeepSeek Flash，不能用仅文本模型替代）。`mem0` 的 embedding 配置独立于聊天供应商；视频和狼人杀的语音服务也独立配置。

固定模型的论文复现实验、历史验证产物、训练脚本、第三方内置项目保留原来的配置和结果。换模型会影响实验指标，新结果需要重新运行，不能沿用已有报告。

## 离线验证

```bash
python -m pip install -e '.[dev]'
python -m pytest -q tests/test_example_providers.py tests/test_providers.py
cd chapter1/web-search-agent
python -m pytest -q
```

测试使用假密钥和模拟 HTTP 传输，检查端点、认证头、模型选择、工具往返及同步/异步流式协议，不调用付费 API。真实账户的模型权限和网络连通性需要使用自己的密钥另行验证。
