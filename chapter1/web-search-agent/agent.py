"""Search with Kimi Formula, DeepSeek Search, or OpenAI Responses web_search."""

# Locate the shared package when this example is run from a source checkout.
import sys as _sys
from pathlib import Path as _CheckoutPath

_sys.path.insert(0, str(_CheckoutPath(__file__).resolve().parents[2]))

import json
from typing import List, Dict, Any, Optional
from openai import APITimeoutError, OpenAI, RateLimitError
from openai.types.chat.chat_completion import Choice
import logging
import requests
import time

# Initialize the checkout path before importing shared provider helpers.
from config import Config
from agentbook.providers import chat_completion, chat_request
from react_trace import (
    STEP_LABELS as STEP_LABELS,
    format_trace_step as format_trace_step,
    run_offline_demo as run_offline_demo,
)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _reasoning_safe_temperature(model, requested=1.0):
    """Reasoning models (Kimi K3, GPT-5, ...) only accept temperature=1.
    Return 1 for those; otherwise the requested value so non-reasoning
    providers (Doubao, DeepSeek, older Moonshot) are unchanged."""
    m = str(model or "").lower().replace("/", "-")
    return 1 if ("kimi-k3" in m or "gpt-5" in m) else requested


# ReAct 轨迹的步骤类型与展示标签（思考 → 行动 → 观察 → 最终答案）




def search_impl(arguments: Dict[str, Any]) -> Any:
    """
    When using the search tool provided by Moonshot AI, you just need to return the arguments as they are,
    without any additional processing logic.
 
    But if you want to use other models and keep the internet search functionality, you just need to modify 
    the implementation here (for example, calling search and fetching web page content), the function signature 
    remains the same and still works.
 
    This ensures maximum compatibility, allowing you to switch between different models without making 
    destructive changes to the code.
    """
    return arguments


# search_and_answer 不抛异常，而是以字符串形式返回失败兜底文案。
# 下列前缀 / 文案是判断“一次搜索是否失败”的唯一来源，供调用方（如
# examples.batch_search）复用，避免把失败响应误判为 success。
SEARCH_ERROR_PREFIX = "搜索过程中出现错误"
MAX_ITERATIONS_MESSAGE = "抱歉，搜索过程超过了最大迭代次数，请稍后重试。"
NO_INFO_MESSAGE = "抱歉，我无法获取足够的信息来回答您的问题。"


def _error_hint(exc: Exception, timeout: Optional[float] = None, provider: str = "kimi") -> str:
    """给 provider 故障补一句可操作的提示。

    交互模式下最常见的两种失败是速率限制和请求超时，而且两者常常连在一起：
    429 会被 SDK 自动重试，重试把超时预算耗完之后，用户只看到一句
    "Request timed out"，很容易误判成网络问题或模型能力问题。
    """
    if isinstance(exc, RateLimitError):
        return "（触发了服务商的速率限制，请降低请求频率或稍后重试）"
    if isinstance(exc, APITimeoutError):
        budget = f"超过 {timeout:.0f} 秒未返回" if timeout else "超时未返回"
        if provider != "kimi":
            return f"（请求{budget}，可调大 SEARCH_TIMEOUT，并检查网络和服务商的速率限制）"
        return (
            f"（请求{budget}。kimi-k3 以 reasoning_effort=max 运行，"
            "单次调用可能需要一到数分钟；若持续超时，可调大环境变量 "
            "SEARCH_TIMEOUT，并检查日志中是否有 429 导致的反复重试）"
        )
    return ""


def is_failure_answer(answer: str) -> bool:
    """判断 search_and_answer 的返回是否为失败兜底（未能正常作答）。"""
    return (
        answer.startswith(SEARCH_ERROR_PREFIX)
        or answer == MAX_ITERATIONS_MESSAGE
        or answer == NO_INFO_MESSAGE
    )


def _render_citations(part: Dict[str, Any]) -> str:
    """Render Responses URL annotations as links without losing answer text."""
    text = part.get("text", "")
    insertions = []
    extra_links = []
    for annotation in part.get("annotations", []):
        if annotation.get("type") != "url_citation":
            continue
        url = annotation.get("url", "")
        if not url.startswith(("https://", "http://")):
            continue
        title = (annotation.get("title") or url).replace("[", "\\[").replace("]", "\\]")
        link = f"[{title}]({url})"
        start, end = annotation.get("start_index"), annotation.get("end_index")
        if isinstance(start, int) and isinstance(end, int) and 0 <= start <= end <= len(text):
            # Some responses expose an opaque citation marker; replace only
            # that marker. When the range covers prose, retain the prose.
            cited = text[start:end]
            if cited.startswith("cite") and cited.endswith(""):
                insertions.append((start, end, link))
            else:
                insertions.append((end, end, f" {link}"))
        elif link not in extra_links:
            extra_links.append(link)
    for start, end, link in sorted(insertions, reverse=True):
        text = text[:start] + link + text[end:]
    if extra_links:
        text += "\n\n" + " · ".join(extra_links)
    return text


class WebSearchAgent:
    """
    Web Search Agent - 使用所选供应商的官方搜索工具。

    kimi-k3 的当前官方路径是标准 ``function`` tool 声明加
    ``moonshot/web-search:latest`` Formula Fiber 执行。
    DeepSeek 通过 function 调用同一密钥的 Search API；OpenAI 通过
    Responses API 在服务端执行 web_search。
    """
    
    def __init__(self, api_key: str = None, base_url: str = None,
                 model: str = None, verbose: bool = False, provider: str = None):
        """
        初始化 Agent

        Args:
            api_key: 所选供应商 API key (如果不提供，从环境变量获取)
            base_url: API 基础 URL
            model: 使用的模型名称（默认 kimi-k3）
            verbose: 是否实时打印 ReAct 轨迹（思考/行动/观察）
        """
        from agentbook.providers import resolve_example_backend
        backend = resolve_example_backend(provider, model, api_key, base_url=base_url)
        resolved_key, resolved_base_url, model, self.using_openrouter = backend
        self.provider = "openrouter" if backend.using_openrouter else backend.provider
        self.client = OpenAI(
            api_key=resolved_key,
            base_url=resolved_base_url,
            # 应用配置的搜索超时，避免后端挂起时请求默认阻塞约 10 分钟
            timeout=Config.SEARCH_TIMEOUT,
        )
        self._api_key = resolved_key
        self.base_url = resolved_base_url
        self.model = model
        self.verbose = verbose
        self.conversation_history = []
        # ReAct 轨迹：按顺序记录每一步的思考/行动/观察，便于展示与调试
        self.trace: List[Dict[str, Any]] = []
        # Credential-free provider requests/responses for Experiment 1-2
        # acceptance.  Search IDs in tool arguments are intentionally retained:
        # they prove that Moonshot's hosted built-in tool actually executed.
        self.api_turns: List[Dict[str, Any]] = []
        self.formula_uri = "moonshot/web-search:latest"
        self._formula_tools: Optional[List[Dict[str, Any]]] = None
        self._request_timeout = Config.SEARCH_TIMEOUT
        self.temperature = 0.6
        # 推理模型（Kimi K3）需要充足的输出预算，避免最终答案被截断
        self.max_tokens = 16384 if self.provider == "openai" else 32768

    def _emit(self, step: Dict[str, Any]):
        """记录一条 ReAct 轨迹步骤，并在 verbose 模式下实时打印。"""
        self.trace.append(step)
        if self.verbose:
            print(format_trace_step(step))
        
    def _get_tools(self) -> List[Dict[str, Any]]:
        """Declare DeepSeek's client tool, or fetch Kimi's Formula declaration."""
        if getattr(self, "provider", "kimi") == "openai":
            return [{"type": "web_search"}]
        if getattr(self, "provider", "kimi") == "deepseek":
            return [{"type": "function", "function": {
                "name": "web_search", "description": "通过 DeepSeek 官方 Search API 搜索网页，返回标题、URL 和正文摘要。",
                "parameters": {"type": "object", "properties": {
                    "query": {"type": "string", "description": "搜索关键词（1–128 字符）",
                              "minLength": 1, "maxLength": 128}
                }, "required": ["query"]}
            }}]
        if getattr(self, "using_openrouter", False):
            return []
        if self._formula_tools is not None:
            return self._formula_tools

        url = (
            f"{self.base_url.rstrip('/')}/formulas/"
            f"{self.formula_uri}/tools"
        )
        started = time.monotonic()
        response = None
        try:
            response = requests.get(
                url,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=self._request_timeout,
            )
            payload = response.json()
            response.raise_for_status()
            tools = payload.get("tools")
            if not isinstance(tools, list) or not tools:
                raise RuntimeError("Formula declaration response has no tools")
            if not any(
                tool.get("type") == "function"
                and tool.get("function", {}).get("name") == "web_search"
                for tool in tools
            ):
                raise RuntimeError(
                    "Formula declaration does not contain function web_search"
                )
        except Exception as exc:
            error_payload: Dict[str, Any] = {
                "class": type(exc).__name__,
                "message": str(exc),
            }
            if response is not None:
                try:
                    error_payload["response"] = response.json()
                except ValueError:
                    error_payload["response_text"] = response.text
            self.api_turns.append({
                "kind": "formula_tools",
                "formula_uri": self.formula_uri,
                "request": {"method": "GET", "url": url},
                "http_status": getattr(response, "status_code", None),
                "elapsed_seconds": round(time.monotonic() - started, 6),
                "error": error_payload,
            })
            raise

        self.api_turns.append({
            "kind": "formula_tools",
            "formula_uri": self.formula_uri,
            "request": {"method": "GET", "url": url},
            "http_status": response.status_code,
            "response": payload,
            "elapsed_seconds": round(time.monotonic() - started, 6),
        })
        self._formula_tools = tools
        return tools

    def _execute_formula(self, name: str, raw_arguments: str) -> str:
        """Execute a client-side search tool (DeepSeek Search or Kimi Fiber)."""
        if getattr(self, "provider", "kimi") == "deepseek":
            return self._search_deepseek(name, raw_arguments)
        if self.using_openrouter:
            raise RuntimeError("Kimi Formula tools are unavailable on OpenRouter")

        url = (
            f"{self.base_url.rstrip('/')}/formulas/"
            f"{self.formula_uri}/fibers"
        )
        body = {"name": name, "arguments": raw_arguments}
        started = time.monotonic()
        response = None
        try:
            response = requests.post(
                url,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=body,
                timeout=self._request_timeout,
            )
            payload = response.json()
            response.raise_for_status()
            if payload.get("status") != "succeeded":
                raise RuntimeError(
                    f"Formula Fiber did not succeed: {payload.get('status')!r}"
                )
            context = payload.get("context") or {}
            result = context.get("output")
            if result in (None, ""):
                result = context.get("encrypted_output")
            if result in (None, ""):
                raise RuntimeError("Succeeded Formula Fiber returned no output")
        except Exception as exc:
            error_payload: Dict[str, Any] = {
                "class": type(exc).__name__,
                "message": str(exc),
            }
            if response is not None:
                try:
                    error_payload["response"] = response.json()
                except ValueError:
                    error_payload["response_text"] = response.text
            self.api_turns.append({
                "kind": "formula_fiber",
                "formula_uri": self.formula_uri,
                "request": {
                    "method": "POST",
                    "url": url,
                    "body": body,
                },
                "http_status": getattr(response, "status_code", None),
                "elapsed_seconds": round(time.monotonic() - started, 6),
                "error": error_payload,
            })
            raise

        self.api_turns.append({
            "kind": "formula_fiber",
            "formula_uri": self.formula_uri,
            "request": {"method": "POST", "url": url, "body": body},
            "http_status": response.status_code,
            "response": payload,
            "elapsed_seconds": round(time.monotonic() - started, 6),
        })
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False)

    def _search_deepseek(self, name: str, raw_arguments: str) -> str:
        """Use the model's own DeepSeek key for the official Search API."""
        arguments = json.loads(raw_arguments)
        query = arguments.get("query") if isinstance(arguments, dict) else None
        if name != "web_search" or not isinstance(query, str) or not query.strip() or len(query) > 128:
            raise ValueError("DeepSeek web_search query 必须是 1–128 字符的非空字符串")
        base = self.base_url.rstrip("/")
        if not base.endswith("/v1"):
            base += "/v1"
        url = f"{base}/tools/search"
        body = {"query": query, "count": 5, "need_content": True}
        receipt = {
            "kind": "deepseek_search",
            "request": {"method": "POST", "url": url, "body": body},
        }
        started = time.monotonic()
        response = None
        try:
            response = requests.post(
                url,
                headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
                json=body,
                timeout=self._request_timeout,
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
                raise RuntimeError("DeepSeek Search response has no results array")
            receipt["response"] = payload
            return json.dumps(payload, ensure_ascii=False)
        except Exception as exc:
            receipt["error"] = {"class": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            receipt["http_status"] = getattr(response, "status_code", None)
            receipt["elapsed_seconds"] = round(time.monotonic() - started, 6)
            self.api_turns.append(receipt)

    def _search_openai(self, user_question: str, max_iterations: int) -> str:
        """Let Responses execute hosted search, then expose its real receipts."""
        if max_iterations <= 0:
            return MAX_ITERATIONS_MESSAGE
        body = {
            "model": self.model,
            "instructions": self._get_system_prompt(),
            "input": user_question,
            "tools": self._get_tools(),
            "tool_choice": "required",
            "max_output_tokens": self.max_tokens,
            "max_tool_calls": max_iterations,
            "include": ["web_search_call.action.sources"],
            "store": False,
        }
        sampling = chat_request(model=self.model, temperature=self.temperature)
        if "temperature" in sampling:
            body["temperature"] = sampling["temperature"]
        # SDK 1.68 supports Responses but predates max_tool_calls. extra_body
        # sends the current wire parameter without raising that SDK's floor.
        kwargs = {key: value for key, value in body.items() if key != "max_tool_calls"}
        receipt = {"kind": "responses", "request": body}
        started = time.monotonic()
        try:
            response = self.client.responses.with_raw_response.create(
                **kwargs, extra_body={"max_tool_calls": max_iterations}
            )
            # Keep the exact JSON, including current tool fields unknown to
            # older SDK response unions, without lossy model reserialization.
            payload = response.http_response.json()
            receipt["response"] = payload
        except Exception as exc:
            receipt["error"] = {"class": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            receipt["elapsed_seconds"] = round(time.monotonic() - started, 6)
            self.api_turns.append(receipt)

        answers = []
        completed_searches = 0
        iteration = 0
        for item in payload.get("output", []):
            if item.get("type") == "web_search_call":
                iteration += 1
                self._emit({"iteration": iteration, "type": "action", "tool": "web_search",
                            "args": item.get("action") or {}})
                self._emit({"iteration": iteration, "type": "observation", "tool": "web_search",
                            "content": json.dumps(item, ensure_ascii=False)})
                if item.get("status") == "completed":
                    completed_searches += 1
            elif item.get("type") == "reasoning":
                for summary in item.get("summary", []):
                    if summary.get("text"):
                        self._emit({"iteration": max(1, iteration), "type": "thought",
                                    "content": summary["text"]})
            elif item.get("type") == "message":
                for part in item.get("content", []):
                    if part.get("type") == "output_text":
                        answers.append(_render_citations(part))

        status = payload.get("status")
        if status not in {"completed", "incomplete"}:
            error = payload.get("error") or {}
            raise RuntimeError(f"OpenAI Responses {status}: {error.get('message', '请求未完成')}")
        if not completed_searches:
            raise RuntimeError("OpenAI 未返回成功的 web_search_call，无法确认实际执行了联网搜索")
        answer = "\n".join(answers).strip()
        if status == "incomplete":
            reason = (payload.get("incomplete_details") or {}).get("reason")
            if reason != "max_output_tokens":
                raise RuntimeError(f"OpenAI Responses 未完成: {reason}")
            note = "（注意：回答因达到 max_output_tokens 上限被截断，请增大 max_tokens 后重试。）"
            answer = f"{answer}\n\n{note}" if answer else note
        if not answer:
            return NO_INFO_MESSAGE
        self._emit({"iteration": max(1, iteration), "type": "answer", "content": answer})
        self.conversation_history.append({"role": "assistant", "content": answer})
        return answer
    
    def _get_system_prompt(self) -> str:
        """
        获取系统提示
        """
        return f"""你是一个智能搜索助手。

请按照以下步骤处理：
1. 分析用户问题，识别关键信息需求
2. 使用 web_search 官方工具搜索相关信息
3. 如果需要更多信息，可以多次调用搜索工具
4. 综合所有信息，生成准确、全面的答案

注意：
- 搜索时使用精准的关键词
- 优先获取最新、最权威的信息
- 答案要结构清晰，有理有据
- 引用搜索结果时给出对应的来源 URL
"""
    
    def _chat(self, messages: List[Dict[str, Any]]) -> Choice:
        """
        调用 Kimi API 进行对话
        
        Args:
            messages: 消息列表
            
        Returns:
            API 响应的 Choice 对象
        """
        kwargs = dict(
            model=self.model,
            messages=messages,
            temperature=_reasoning_safe_temperature(self.model, self.temperature),
            # Kimi K3 是推理模型，会先产出较长的 reasoning_content，需要给最终回答
            # 留足输出预算（Moonshot 要求 max_tokens>=2048），否则答案可能被截断为空。
            max_tokens=self.max_tokens,
        )
        if str(self.model).lower() == "kimi-k3":
            kwargs["reasoning_effort"] = "max"
        tools = self._get_tools()
        if tools:  # OpenRouter 兜底时无内置搜索工具，省略 tools 参数
            kwargs["tools"] = tools
        started = time.monotonic()
        try:
            kwargs = chat_request(**kwargs)
            completion = chat_completion(self.client, **kwargs)
        except Exception as exc:
            self.api_turns.append({
                "kind": "chat_completion",
                "request": json.loads(json.dumps(kwargs, ensure_ascii=False, default=str)),
                "elapsed_seconds": round(time.monotonic() - started, 6),
                "error": {"class": type(exc).__name__, "message": str(exc)},
            })
            raise
        response = (
            completion.model_dump() if hasattr(completion, "model_dump")
            else completion.dict() if hasattr(completion, "dict")
            else {"raw_response": str(completion)}
        )
        self.api_turns.append({
            "kind": "chat_completion",
            "request": json.loads(json.dumps(kwargs, ensure_ascii=False, default=str)),
            "response": json.loads(json.dumps(response, ensure_ascii=False, default=str)),
            "elapsed_seconds": round(time.monotonic() - started, 6),
        })
        return completion.choices[0]

    def search_and_answer(self, user_question: str, max_iterations: int = 5) -> str:
        """
        执行搜索并生成答案
        
        Args:
            user_question: 用户问题
            max_iterations: 最大搜索迭代次数（防止无限循环）
            
        Returns:
            最终答案
        """
        # 构建系统提示
        system_prompt = self._get_system_prompt()
        
        # 重置对话历史并添加新的系统提示
        self.conversation_history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question}
        ]
        # 重置 ReAct 轨迹
        self.trace = []
        self.api_turns = []
        # Each independent question keeps its own real declaration receipt.
        self._formula_tools = None
        logger.info("开始调用 %s 搜索工具...", getattr(self, "provider", "kimi"))

        try:
            if getattr(self, "provider", "kimi") == "openai":
                return self._search_openai(user_question, max_iterations)
            finish_reason = None
            iteration = 0
            
            # 循环处理，直到获得最终答案或达到最大迭代次数
            while (finish_reason is None or finish_reason == "tool_calls") and iteration < max_iterations:
                iteration += 1
                logger.info(f"迭代 {iteration}/{max_iterations}")
                
                # 调用 Kimi API
                choice = self._chat(self.conversation_history)
                finish_reason = choice.finish_reason
                
                # 捕获模型的思考过程（Kimi K3 等推理模型通过 reasoning_content 暴露思考模式）
                reasoning = getattr(choice.message, "reasoning_content", None)
                if reasoning:
                    self._emit({"iteration": iteration, "type": "thought", "content": reasoning})

                if finish_reason == "tool_calls":
                    # 处理工具调用
                    logger.info(f"模型请求调用 {len(choice.message.tool_calls)} 个工具")

                    # 添加助手的消息（包含工具调用）到历史。
                    # 注意：必须把消息重建为纯 dict，而不是直接塞入 SDK 返回的
                    # pydantic message 对象——后者会附带 reasoning_content / refusal
                    # 等额外字段，回传给 Moonshot 时会触发 "tokenization failed" 400 错误。
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": choice.message.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in choice.message.tool_calls
                        ],
                    })

                    # 执行每个工具调用
                    for tool_call in choice.message.tool_calls:
                        tool_call_name = tool_call.function.name
                        try:
                            tool_call_arguments = json.loads(
                                tool_call.function.arguments or "{}"
                            )
                        except json.JSONDecodeError:
                            # Models sometimes emit slightly invalid JSON; match
                            # chapter4 async-agent and keep the ReAct loop alive.
                            tool_call_arguments = {}
                            logger.warning(
                                "工具参数不是合法 JSON，已按空对象继续: %r",
                                tool_call.function.arguments,
                            )

                        logger.info(f"执行工具: {tool_call_name}, 参数: {tool_call_arguments}")
                        # 行动：记录一次工具调用
                        self._emit({"iteration": iteration, "type": "action",
                                    "tool": tool_call_name, "args": tool_call_arguments})

                        if tool_call_name == "web_search":
                            # Formula requires the original serialized
                            # arguments, even though the parsed copy above is
                            # retained for a readable ReAct trace.
                            tool_result = self._execute_formula(
                                tool_call_name,
                                tool_call.function.arguments or "{}",
                            )
                        else:
                            tool_result = f"Error: unable to find tool by name '{tool_call_name}'"

                        tool_content = (
                            tool_result
                            if isinstance(tool_result, str)
                            else json.dumps(tool_result, ensure_ascii=False)
                        )
                        # 观察：记录工具返回结果
                        self._emit({"iteration": iteration, "type": "observation",
                                    "tool": tool_call_name, "content": tool_content})
                        # 构建工具响应消息并添加到历史
                        self.conversation_history.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_content
                        })
                elif finish_reason == "length":
                    # 输出预算（max_tokens）耗尽导致截断：返回已生成内容并明确标注，
                    # 而不是把半截答案当作完整答案，也不误报“无法获取足够信息”
                    # （content 为空时，思考过程已耗尽整个预算）。
                    partial = (choice.message.content or "").strip()
                    logger.warning("回答因达到 max_tokens 上限被截断 (finish_reason=length)")
                    note = "（注意：回答因达到 max_tokens 上限被截断，请增大 max_tokens 后重试。）"
                    final = f"{partial}\n\n{note}" if partial else note
                    self._emit({"iteration": iteration, "type": "answer", "content": final})
                    # 存入历史时保留截断提示（final），否则 get_conversation_history()
                    # 会丢失截断语义，后续复用历史时可能把不完整回答当作普通回答。
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": final
                    })
                    return final
                else:
                    # 获得最终答案
                    if choice.message.content:
                        answer = choice.message.content
                        logger.info("成功生成答案")
                        self._emit({"iteration": iteration, "type": "answer", "content": answer})

                        # 添加最终答案到历史
                        self.conversation_history.append({
                            "role": "assistant",
                            "content": answer
                        })

                        return answer
            
            # 如果达到最大迭代次数仍未完成
            if iteration >= max_iterations:
                logger.warning(f"达到最大迭代次数 {max_iterations}")
                return MAX_ITERATIONS_MESSAGE

            return NO_INFO_MESSAGE
                
        except Exception as e:
            hint = _error_hint(e, getattr(self, "_request_timeout", None), getattr(self, "provider", "kimi"))
            message = f"{SEARCH_ERROR_PREFIX}: {str(e)}{hint}"
            logger.error(message)
            return message
    
    def clear_history(self):
        """清空对话历史"""
        self.conversation_history = []
        logger.info("对话历史已清空")
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """获取对话历史"""
        return self.conversation_history

    def get_trace(self) -> List[Dict[str, Any]]:
        """获取上一次 search_and_answer 的 ReAct 轨迹（思考/行动/观察/最终答案）"""
        return self.trace

    def get_api_turns(self) -> List[Dict[str, Any]]:
        """Return detached real-provider evidence for the latest question."""
        return json.loads(json.dumps(self.api_turns, ensure_ascii=False, default=str))
    
    def set_temperature(self, temperature: float):
        """
        设置温度参数
        
        Args:
            temperature: 温度值 (0.0 - 2.0)
        """
        if 0.0 <= temperature <= 2.0:
            self.temperature = temperature
            logger.info(f"温度设置为: {temperature}")
        else:
            logger.warning(f"无效的温度值: {temperature}，应在 0.0 到 2.0 之间")




# 独立运行示例
def main():
    """
    独立运行示例，演示基本用法
    """
    # 设置 API key (确保已设置环境变量 MOONSHOT_API_KEY)
    agent = WebSearchAgent()
    
    # 示例问题
    test_question = "请搜索 Moonshot AI Context Caching 技术，告诉我这是什么。"
    
    print(f"问题: {test_question}")
    print("-" * 60)
    print("搜索中...")
    
    # 获取答案
    answer = agent.search_and_answer(test_question)
    
    print("\n答案:")
    print("-" * 60)
    print(answer)


if __name__ == '__main__':
    main()
