# -*- coding: utf-8 -*-
"""离线自测：不需要安装 AstrBot / aiohttp，也不会访问网络。

做法是用桩模块替换 ``astrbot`` 与 ``aiohttp``，再用 importlib 加载 ``main.py``，
从而在没有 AstrBot 运行时的机器上验证插件的全部逻辑。

覆盖范围：

* 纯函数：指令解析、尺寸归一化、占位符替换、JSON 路径取值、图片探测；
* 三类接口：OpenAI 兼容 / SD WebUI / 自定义，检查请求 URL、请求头、请求体与响应解析；
* 异常路径：HTTP 错误、非 JSON 响应、网络错误重试；
* 插件层：指令出图、图片落盘、单用户冷却、帮助文本、大模型工具、生命周期。

运行： ``python tests/test_offline.py``
"""

from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
MAIN_PY = ROOT / "main.py"

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmM"
    "IQAAAABJRU5ErkJggg=="
)
PNG_B64 = base64.b64encode(PNG_BYTES).decode()
PNG_B64_DATA_URL = "data:image/png;base64," + PNG_B64


# --------------------------------------------------------------------------- #
# 假 aiohttp：记录请求、返回预设响应，全程不碰网络
# --------------------------------------------------------------------------- #
class FakeClientError(Exception):
    """对应 aiohttp.ClientError。"""


class FakeTimeout:
    def __init__(self, total: float | None = None) -> None:
        self.total = total


class _FakeContent:
    def __init__(self, data: bytes) -> None:
        self._data = data

    async def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            return self._data
        return self._data[:n]


class _FakeResponse:
    def __init__(self, status: int, text: str, content: bytes) -> None:
        self.status = status
        self._text = text
        self._content = content
        self.content_length = len(content)
        self.content = _FakeContent(content)

    async def text(self) -> str:
        return self._text


class _RequestContext:
    """模拟 ``async with session.request(...) as resp``。"""

    def __init__(self, factory: Callable[[], Any]) -> None:
        self._factory = factory
        self._resp: _FakeResponse | None = None

    async def __aenter__(self) -> _FakeResponse:
        result = self._factory()
        if asyncio.iscoroutine(result):
            result = await result
        self._resp = result
        return result

    async def __aexit__(self, *exc_info: Any) -> bool:
        return False


class FakeHTTPServer:
    """所有假请求的处理中枢。"""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.handler: Callable[[dict[str, Any]], Any] | None = None

    def reset(self, handler: Callable[[dict[str, Any]], Any] | None = None) -> None:
        self.calls.clear()
        self.handler = handler

    async def handle(self, method: str, url: str, kwargs: dict[str, Any]) -> _FakeResponse:
        call = {
            "method": method.upper(),
            "url": url,
            "headers": dict(kwargs.get("headers") or {}),
            "json": kwargs.get("json"),
            "params": kwargs.get("params"),
        }
        self.calls.append(call)

        if self.handler is None:
            return _FakeResponse(200, "{}", b"{}")

        result = self.handler(call)
        if asyncio.iscoroutine(result):
            result = await result
        if isinstance(result, BaseException):
            raise result
        if isinstance(result, _FakeResponse):
            return result
        if isinstance(result, bytes):
            return _FakeResponse(200, result.decode("utf-8", "replace"), result)

        status, text = result[0], result[1]
        content = result[2] if len(result) > 2 else str(text).encode("utf-8")
        return _FakeResponse(int(status), str(text), content)


class FakeClientSession:
    def __init__(self, timeout: Any = None, **kwargs: Any) -> None:
        self.timeout = timeout
        self.closed = False
        self.server: FakeHTTPServer = HTTP

    async def close(self) -> None:
        self.closed = True

    def request(self, method: str, url: str, **kwargs: Any) -> _RequestContext:
        return _RequestContext(lambda: self.server.handle(method, url, kwargs))

    def get(self, url: str, **kwargs: Any) -> _RequestContext:
        return _RequestContext(lambda: self.server.handle("GET", url, kwargs))

    async def __aenter__(self) -> "FakeClientSession":
        return self

    async def __aexit__(self, *exc_info: Any) -> bool:
        await self.close()
        return False


HTTP = FakeHTTPServer()


def install_aiohttp_stub() -> types.ModuleType:
    module = types.ModuleType("aiohttp")
    module.ClientError = FakeClientError
    module.ClientTimeout = FakeTimeout
    module.ClientSession = FakeClientSession
    module.ClientResponseError = FakeClientError
    sys.modules["aiohttp"] = module
    return module


# --------------------------------------------------------------------------- #
# 假 AstrBot
# --------------------------------------------------------------------------- #
class FakeLogger:
    def __init__(self) -> None:
        self.records: list[tuple[str, str]] = []

    def _log(self, level: str, message: str) -> None:
        self.records.append((level, str(message)))

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._log("debug", message)

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._log("info", message)

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._log("warning", message)

    warn = warning

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._log("error", message)


LOGGER = FakeLogger()
REGISTRY: dict[str, list[str]] = {"commands": [], "llm_tools": []}


class FakeFilter:
    @staticmethod
    def command(name: str, *args: Any, **kwargs: Any) -> Callable[[Any], Any]:
        def decorator(func: Any) -> Any:
            REGISTRY["commands"].append(name)
            return func

        return decorator

    @staticmethod
    def llm_tool(name: str | None = None, *args: Any, **kwargs: Any) -> Callable[[Any], Any]:
        def decorator(func: Any) -> Any:
            REGISTRY["llm_tools"].append(name or getattr(func, "__name__", ""))
            return func

        return decorator


class FakeEvent:
    def __init__(self, message: str = "", sender: str = "user-1") -> None:
        self.message_str = message
        self._sender = sender

    def get_sender_id(self) -> str:
        return self._sender

    def plain_result(self, text: str) -> Any:
        return types.SimpleNamespace(kind="plain", text=text)

    def image_result(self, path: str) -> Any:
        return types.SimpleNamespace(kind="image", path=path)

    def chain_result(self, chain: Any) -> Any:
        return types.SimpleNamespace(kind="chain", chain=chain)


def install_astrbot_stub(data_dir: Path) -> types.ModuleType:
    astrbot = types.ModuleType("astrbot")
    astrbot.__path__ = []  # type: ignore[attr-defined]

    api = types.ModuleType("astrbot.api")
    api.__path__ = []  # type: ignore[attr-defined]

    class AstrBotConfig(dict):  # noqa: D101 - 桩
        pass

    api.AstrBotConfig = AstrBotConfig
    api.logger = LOGGER

    event_mod = types.ModuleType("astrbot.api.event")
    event_mod.AstrMessageEvent = FakeEvent
    event_mod.filter = FakeFilter
    event_mod.AstrBotEventBase = object

    star_mod = types.ModuleType("astrbot.api.star")

    class Context:  # noqa: D101 - 桩
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args

    class Star:  # noqa: D101 - 桩
        def __init__(self, context: Any = None, *args: Any, **kwargs: Any) -> None:
            self.context = context

    class StarTools:
        @staticmethod
        def get_data_dir(name: str) -> str:
            target = data_dir / name
            target.mkdir(parents=True, exist_ok=True)
            return str(target)

    star_mod.Context = Context
    star_mod.Star = Star
    star_mod.StarTools = StarTools

    api.event = event_mod
    api.star = star_mod
    astrbot.api = api

    sys.modules["astrbot"] = astrbot
    sys.modules["astrbot.api"] = api
    sys.modules["astrbot.api.event"] = event_mod
    sys.modules["astrbot.api.star"] = star_mod
    return astrbot


def load_main(data_dir: Path) -> types.ModuleType:
    install_aiohttp_stub()
    install_astrbot_stub(data_dir)
    spec = importlib.util.spec_from_file_location("image_api_under_test", MAIN_PY)
    assert spec and spec.loader, "无法加载 main.py"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# 用例注册
# --------------------------------------------------------------------------- #
CASES: list[tuple[str, Callable[[], Any]]] = []


def case(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        CASES.append((name, func))
        return func

    return decorator


def base_config() -> dict[str, Any]:
    return {
        "provider": "openai",
        "api_base": "https://api.example.com/v1",
        "api_key": "test-key",
        "auth_mode": "bearer",
        "model": "test-model",
        "size": "512x512",
        "default_n": 1,
        "max_n": 4,
        "response_format": "auto",
        "quality": "",
        "extra_headers": {},
        "extra_body": {},
        "timeout": 30,
        "retry": 0,
        "max_concurrency": 2,
        "cooldown_seconds": 10,
        "max_prompt_length": 800,
        "notify_before_generate": True,
        "keep_days": 7,
        "enable_llm_tool": True,
        "sd_base_url": "http://sd.local:7860",
        "sd_negative_prompt": "lowres",
        "sd_steps": 12,
        "sd_cfg_scale": 5.5,
        "sd_sampler": "Euler a",
        "custom_url": "",
        "custom_method": "POST",
        "custom_body": "",
        "custom_image_path": "",
    }


def new_client(main: types.ModuleType, **overrides: Any) -> Any:
    config = base_config()
    config.update(overrides)
    return main.ImageAPIClient(config)


def new_plugin(main: types.ModuleType, **overrides: Any) -> Any:
    config = base_config()
    config.update(overrides)
    plugin = main.ImageAPIPlugin(context=object(), config=config)
    return plugin


# --------------------------------------------------------------------------- #
# 1. 纯函数
# --------------------------------------------------------------------------- #
@case("normalize_size / parse_size")
def test_size(main: types.ModuleType) -> None:
    assert main.parse_size("1024x768") == (1024, 768)
    assert main.parse_size("512 * 512") == (512, 512)
    assert main.parse_size("bad") is None
    assert main.normalize_size("1024X1792") == "1024x1792"
    assert main.normalize_size("nope") == ""


@case("detect_image_ext")
def test_detect_ext(main: types.ModuleType) -> None:
    assert main.detect_image_ext(PNG_BYTES) == ".png"
    assert main.detect_image_ext(b"\xff\xd8\xff\xe0rest") == ".jpg"
    assert main.detect_image_ext(b"RIFF1234WEBPVP8 ") == ".webp"
    assert main.detect_image_ext(b"GIF89a....") == ".gif"
    assert main.detect_image_ext(b"unknown") == ".png"


@case("parse_command_text: 基本与参数")
def test_parse_command(main: types.ModuleType) -> None:
    parsed = main.parse_command_text("/画图 一只在月球上钓鱼的猫 --size 768x768 --n 2")
    assert parsed.prompt == "一只在月球上钓鱼的猫", parsed.prompt
    assert parsed.size == "768x768"
    assert parsed.n == 2
    assert not parsed.help_requested

    parsed = main.parse_command_text('画图 水墨山水 --model flux --neg "模糊, 低质量" --seed 42')
    assert parsed.prompt == "水墨山水", parsed.prompt
    assert parsed.model == "flux"
    assert parsed.negative_prompt == "模糊, 低质量", parsed.negative_prompt
    assert parsed.seed == 42

    parsed = main.parse_command_text("draw a cat --size=1024x1024")
    assert parsed.prompt == "a cat", parsed.prompt
    assert parsed.size == "1024x1024"


@case("parse_command_text: 帮助与空输入")
def test_parse_help(main: types.ModuleType) -> None:
    for text in ("/画图", "画图 帮助", "/画图 --help", "文生图"):
        assert main.parse_command_text(text).help_requested, text
    assert not main.parse_command_text("/画图 猫").help_requested


@case("replace_placeholders: URL 与 JSON 两种模式")
def test_placeholders(main: types.ModuleType) -> None:
    values = {"prompt": "猫 & 狗", "n": "2", "unknown_left": ""}
    url = main.replace_placeholders("https://x/gen?q={prompt}&n={n}", values, url_mode=True)
    assert url == "https://x/gen?q=%E7%8C%AB%20%26%20%E7%8B%97&n=2", url

    body = main.replace_placeholders('{"q": "{prompt}"}', values, url_mode=False)
    assert body == '{"q": "猫 & 狗"}', body
    assert json.loads(body)["q"] == "猫 & 狗"

    quoted = main.replace_placeholders('{"q": "{prompt}"}', {"prompt": 'a"b'}, url_mode=False)
    assert json.loads(quoted)["q"] == 'a"b'

    # 未知占位符原样保留
    assert main.replace_placeholders("{other}", values, url_mode=True) == "{other}"


@case("json_path_get / auto_find_image")
def test_json_helpers(main: types.ModuleType) -> None:
    payload = {"result": {"images": [{"url": "https://cdn/a.png"}]}}
    assert main.json_path_get(payload, "result.images.0.url") == "https://cdn/a.png"
    assert main.json_path_get(payload, "result.images.5.url") is None
    assert main.json_path_get(payload, "missing.x") is None
    assert main.json_path_get(payload, "") is None

    found = main.auto_find_image(payload)
    assert found and found.url == "https://cdn/a.png"

    long_b64 = PNG_B64 * 8  # 超过 512 字符才会被当成 base64 候选
    found = main.auto_find_image({"data": [{"b64_json": long_b64}]})
    assert found and not found.url and found.b64 == long_b64

    assert main.auto_find_image({"nothing": 1}) is None


@case("format_http_error 中文提示")
def test_format_http_error(main: types.ModuleType) -> None:
    message = main.format_http_error(401, '{"error": {"message": "invalid key"}}')
    assert "401" in message and "API Key" in message and "invalid key" in message, message

    message = main.format_http_error(429, "rate limited")
    assert "429" in message and "频繁" in message, message


# --------------------------------------------------------------------------- #
# 2. 客户端：OpenAI 兼容
# --------------------------------------------------------------------------- #
@case("openai: b64 返回 + 请求构造")
def test_openai_b64(main: types.ModuleType) -> None:
    async def run() -> None:
        HTTP.reset(lambda call: (200, json.dumps({"data": [{"b64_json": PNG_B64, "revised_prompt": "更详细的猫"}]})))
        client = new_client(main)
        try:
            results = await client.generate(main.GenRequest(prompt="一只猫", size="512x512", n=2))
        finally:
            await client.close()

        assert len(results) == 1
        data, revised = results[0]
        assert data == PNG_BYTES
        assert revised == "更详细的猫"

        call = HTTP.calls[0]
        assert call["method"] == "POST"
        assert call["url"] == "https://api.example.com/v1/images/generations", call["url"]
        assert call["headers"]["Authorization"] == "Bearer test-key"
        assert call["headers"]["Accept"] == "application/json"
        body = call["json"]
        assert body["model"] == "test-model"
        assert body["prompt"] == "一只猫"
        assert body["n"] == 2
        assert body["size"] == "512x512"
        assert "response_format" not in body, "response_format=auto 时不应发送该字段"
        assert "quality" not in body

    asyncio.run(run())


@case("openai: url 返回 + 下载图片")
def test_openai_url(main: types.ModuleType) -> None:
    async def run() -> None:
        def handler(call: dict[str, Any]) -> Any:
            if call["method"] == "GET":
                return (200, "", PNG_BYTES)
            return (200, json.dumps({"data": [{"url": "https://cdn.example.com/a.png"}]}))

        HTTP.reset(handler)
        client = new_client(main)
        try:
            results = await client.generate(main.GenRequest(prompt="猫"))
        finally:
            await client.close()

        assert len(results) == 1 and results[0][0] == PNG_BYTES
        assert HTTP.calls[1]["method"] == "GET"
        assert HTTP.calls[1]["url"] == "https://cdn.example.com/a.png"

    asyncio.run(run())


@case("openai: data URL 形式的 base64")
def test_openai_data_url(main: types.ModuleType) -> None:
    async def run() -> None:
        HTTP.reset(lambda call: (200, json.dumps({"data": [{"b64_json": PNG_B64_DATA_URL}]})))
        client = new_client(main)
        try:
            results = await client.generate(main.GenRequest(prompt="猫"))
        finally:
            await client.close()
        assert results[0][0] == PNG_BYTES

    asyncio.run(run())


@case("openai: 自动兜底识别非标准结构")
def test_openai_autofind(main: types.ModuleType) -> None:
    async def run() -> None:
        def handler(call: dict[str, Any]) -> Any:
            if call["method"] == "GET":
                return (200, "", PNG_BYTES)
            return (200, json.dumps({"output": [{"image": "https://cdn.example.com/b.png"}]}))

        HTTP.reset(handler)
        client = new_client(main)
        try:
            results = await client.generate(main.GenRequest(prompt="猫"))
        finally:
            await client.close()
        assert results[0][0] == PNG_BYTES

    asyncio.run(run())


@case("openai: HTTP 401 报错友好")
def test_openai_401(main: types.ModuleType) -> None:
    async def run() -> None:
        HTTP.reset(lambda call: (401, json.dumps({"error": {"message": "invalid api key"}})))
        client = new_client(main)
        try:
            await client.generate(main.GenRequest(prompt="猫"))
        except main.ImageGenError as exc:
            assert "401" in str(exc) and "API Key" in str(exc), str(exc)
        else:
            raise AssertionError("应当抛出 ImageGenError")
        finally:
            await client.close()

    asyncio.run(run())


@case("openai: 非 JSON 响应报错")
def test_openai_bad_json(main: types.ModuleType) -> None:
    async def run() -> None:
        HTTP.reset(lambda call: (200, "<html>oops</html>"))
        client = new_client(main)
        try:
            await client.generate(main.GenRequest(prompt="猫"))
        except main.ImageGenError as exc:
            assert "JSON" in str(exc), str(exc)
        else:
            raise AssertionError("应当抛出 ImageGenError")
        finally:
            await client.close()

    asyncio.run(run())


@case("openai: 空 data 报错")
def test_openai_empty(main: types.ModuleType) -> None:
    async def run() -> None:
        HTTP.reset(lambda call: (200, json.dumps({"data": []})))
        client = new_client(main)
        try:
            await client.generate(main.GenRequest(prompt="猫"))
        except main.ImageGenError as exc:
            assert "没有返回任何图片" in str(exc), str(exc)
        else:
            raise AssertionError("应当抛出 ImageGenError")
        finally:
            await client.close()

    asyncio.run(run())


@case("openai: 网络错误重试后成功")
def test_retry(main: types.ModuleType) -> None:
    async def run() -> None:
        state = {"count": 0}

        def handler(call: dict[str, Any]) -> Any:
            state["count"] += 1
            if state["count"] == 1:
                return FakeClientError("connection reset")
            return (200, json.dumps({"data": [{"b64_json": PNG_B64}]}))

        HTTP.reset(handler)
        client = new_client(main, retry=1)

        real_sleep = asyncio.sleep
        asyncio.sleep = lambda *_a, **_k: real_sleep(0);  # type: ignore[assignment]
        try:
            results = await client.generate(main.GenRequest(prompt="猫"))
        finally:
            asyncio.sleep = real_sleep  # type: ignore[assignment]
            await client.close()

        assert state["count"] == 2
        assert results[0][0] == PNG_BYTES

    asyncio.run(run())


@case("openai: 重试耗尽后报网络错误")
def test_retry_exhausted(main: types.ModuleType) -> None:
    async def run() -> None:
        HTTP.reset(lambda call: FakeClientError("boom"))
        client = new_client(main, retry=0)
        try:
            await client.generate(main.GenRequest(prompt="猫"))
        except main.ImageGenError as exc:
            assert "网络请求失败" in str(exc), str(exc)
        else:
            raise AssertionError("应当抛出 ImageGenError")
        finally:
            await client.close()

    asyncio.run(run())


@case("鉴权：api-key 请求头与 none")
def test_auth_modes(main: types.ModuleType) -> None:
    assert new_client(main, auth_mode="api-key", api_key_header="X-Token")._headers()["X-Token"] == "test-key"
    assert "Authorization" not in new_client(main, auth_mode="none")._headers()
    headers = new_client(main, extra_headers={"X-Extra": "1"})._headers()
    assert headers["X-Extra"] == "1"


@case("_openai_url: 完整地址与 base 拼接")
def test_openai_url(main: types.ModuleType) -> None:
    assert new_client(main)._openai_url() == "https://api.example.com/v1/images/generations"
    full = "https://api.example.com/v1/images/generations"
    assert new_client(main, api_base=full)._openai_url() == full
    assert new_client(main, api_base="https://api.example.com/v1/")._openai_url() == full


# --------------------------------------------------------------------------- #
# 3. 客户端：SD WebUI / 自定义
# --------------------------------------------------------------------------- #
@case("sdwebui: 请求体与响应解析")
def test_sdwebui(main: types.ModuleType) -> None:
    async def run() -> None:
        HTTP.reset(lambda call: (200, json.dumps({"images": [PNG_B64, PNG_B64]})))
        client = new_client(main, provider="sdwebui")
        try:
            results = await client.generate(
                main.GenRequest(prompt="赛博朋克", size="640x960", n=2, seed=1234)
            )
        finally:
            await client.close()

        assert len(results) == 2 and results[0][0] == PNG_BYTES

        call = HTTP.calls[0]
        assert call["url"] == "http://sd.local:7860/sdapi/v1/txt2img", call["url"]
        body = call["json"]
        assert body["prompt"] == "赛博朋克"
        assert body["negative_prompt"] == "lowres"
        assert (body["width"], body["height"]) == (640, 960)
        assert body["batch_size"] == 2
        assert body["steps"] == 12
        assert body["cfg_scale"] == 5.5
        assert body["sampler_name"] == "Euler a"
        assert body["seed"] == 1234

    asyncio.run(run())


@case("sdwebui: 未配置地址时报错")
def test_sdwebui_no_base(main: types.ModuleType) -> None:
    async def run() -> None:
        HTTP.reset(None)
        client = new_client(main, provider="sdwebui", sd_base_url="")
        try:
            await client.generate(main.GenRequest(prompt="x"))
        except main.ImageGenError as exc:
            assert "sd_base_url" in str(exc), str(exc)
        else:
            raise AssertionError("应当抛出 ImageGenError")
        finally:
            await client.close()

    asyncio.run(run())


@case("custom: 占位符替换 + JSON 路径取值")
def test_custom_post(main: types.ModuleType) -> None:
    async def run() -> None:
        def handler(call: dict[str, Any]) -> Any:
            if call["method"] == "GET":
                return (200, "", PNG_BYTES)
            return (200, json.dumps({"result": {"images": [{"url": "https://cdn.example.com/c.png"}]}}))

        HTTP.reset(handler)
        client = new_client(
            main,
            provider="custom",
            custom_url="https://api.custom.com/gen?q={prompt}&size={size}",
            custom_body='{"text": "{prompt}", "num": {n}, "w": {width}, "h": {height}}',
            custom_image_path="result.images.0.url",
        )
        try:
            results = await client.generate(main.GenRequest(prompt="猫 狗", size="640x960", n=3))
        finally:
            await client.close()

        assert results[0][0] == PNG_BYTES

        first = HTTP.calls[0]
        assert first["url"].startswith("https://api.custom.com/gen?q=")
        assert "%E7%8C%AB%20%E7%8B%97" in first["url"], first["url"]
        assert "size=640x960" in first["url"], first["url"]
        body = first["json"]
        assert body == {"text": "猫 狗", "num": 3, "w": 640, "h": 960}, body

    asyncio.run(run())


@case("custom: GET 请求走 params")
def test_custom_get(main: types.ModuleType) -> None:
    async def run() -> None:
        def handler(call: dict[str, Any]) -> Any:
            if call["method"] == "GET" and "cdn" in call["url"]:
                return (200, "", PNG_BYTES)
            return (200, json.dumps({"image": "https://cdn.example.com/d.png"}))

        HTTP.reset(handler)
        client = new_client(
            main,
            provider="custom",
            custom_url="https://api.custom.com/gen",
            custom_method="GET",
            custom_body='{"text": "{prompt}"}',
        )
        try:
            results = await client.generate(main.GenRequest(prompt="猫"))
        finally:
            await client.close()

        assert results[0][0] == PNG_BYTES
        assert HTTP.calls[0]["method"] == "GET"
        assert HTTP.calls[0]["params"] == {"text": "猫"}

    asyncio.run(run())


@case("custom: 未配置地址 / 非法 JSON 报错")
def test_custom_errors(main: types.ModuleType) -> None:
    async def run() -> None:
        HTTP.reset(None)
        client = new_client(main, provider="custom", custom_url="")
        try:
            await client.generate(main.GenRequest(prompt="x"))
        except main.ImageGenError as exc:
            assert "custom_url" in str(exc), str(exc)
        else:
            raise AssertionError("应当抛出 ImageGenError")
        finally:
            await client.close()

        HTTP.reset(None)
        client = new_client(
            main, provider="custom", custom_url="https://api.custom.com/gen", custom_body='{"text": "{prompt}"'
        )
        try:
            await client.generate(main.GenRequest(prompt="x"))
        except main.ImageGenError as exc:
            assert "custom_body" in str(exc), str(exc)
        else:
            raise AssertionError("应当抛出 ImageGenError")
        finally:
            await client.close()

    asyncio.run(run())


# --------------------------------------------------------------------------- #
# 4. 插件层
# --------------------------------------------------------------------------- #
@case("插件注册了指令与大模型工具")
def test_registry(main: types.ModuleType) -> None:
    assert set(REGISTRY["commands"]) >= {"画图", "绘图", "文生图", "draw"}, REGISTRY
    assert "generate_image" in REGISTRY["llm_tools"], REGISTRY
    assert hasattr(main, "ImageAPIPlugin")
    doc = main.ImageAPIPlugin.generate_image_tool.__doc__ or ""
    assert "Args:" in doc and "prompt(string)" in doc, doc


@case("插件出图：通知 + 图片 + 汇总，并落盘")
def test_plugin_draw(main: types.ModuleType) -> None:
    async def run() -> None:
        HTTP.reset(lambda call: (200, json.dumps({"data": [{"b64_json": PNG_B64}] * 2})))
        plugin = new_plugin(main)
        event = FakeEvent("/画图 一只猫 --n 2")

        results = [r async for r in plugin._handle_draw(event)]
        kinds = [r.kind for r in results]
        assert kinds == ["plain", "image", "image", "plain"], kinds
        assert results[0].text.startswith("🎨")
        assert "已生成 2 张图片" in results[-1].text, results[-1].text

        saved = [Path(r.path) for r in results[1:3]]
        for path in saved:
            assert path.is_file(), path
            assert path.suffix == ".png"
            assert path.read_bytes() == PNG_BYTES
            assert str(plugin.images_dir) in str(path)

        body = HTTP.calls[0]["json"]
        assert body["n"] == 2

        await plugin.terminate()

    asyncio.run(run())


@case("插件：帮助文本与空提示词")
def test_plugin_help(main: types.ModuleType) -> None:
    async def run() -> None:
        HTTP.reset(None)
        plugin = new_plugin(main)
        for text in ("/画图 帮助", "/画图"):
            results = [r async for r in plugin._handle_draw(FakeEvent(text))]
            assert len(results) == 1 and results[0].kind == "plain"
            assert "图片生成插件使用说明" in results[0].text
        assert not HTTP.calls, "帮助分支不应发起请求"
        await plugin.terminate()

    asyncio.run(run())


@case("插件：单用户冷却")
def test_plugin_cooldown(main: types.ModuleType) -> None:
    async def run() -> None:
        HTTP.reset(lambda call: (200, json.dumps({"data": [{"b64_json": PNG_B64}]})))
        plugin = new_plugin(main, cooldown_seconds=30)
        event = FakeEvent("/画图 猫")

        first = [r async for r in plugin._handle_draw(event)]
        assert any(r.kind == "image" for r in first)

        second = [r async for r in plugin._handle_draw(FakeEvent("/画图 猫"))]
        assert len(second) == 1 and second[0].kind == "plain"
        assert "请求太频繁" in second[0].text, second[0].text

        # 其他用户不受影响
        third = [r async for r in plugin._handle_draw(FakeEvent("/画图 猫", sender="user-2"))]
        assert any(r.kind == "image" for r in third)

        await plugin.terminate()

    asyncio.run(run())


@case("插件：提示词过长")
def test_plugin_too_long(main: types.ModuleType) -> None:
    async def run() -> None:
        HTTP.reset(None)
        plugin = new_plugin(main, max_prompt_length=10)
        results = [r async for r in plugin._handle_draw(FakeEvent("/画图 " + "猫" * 20))]
        assert len(results) == 1 and "太长" in results[0].text, results[0].text
        assert not HTTP.calls
        await plugin.terminate()

    asyncio.run(run())


@case("插件：生成失败时不崩溃")
def test_plugin_error(main: types.ModuleType) -> None:
    async def run() -> None:
        HTTP.reset(lambda call: (403, json.dumps({"error": {"message": "forbidden"}})))
        plugin = new_plugin(main)
        results = [r async for r in plugin._handle_draw(FakeEvent("/画图 猫"))]
        assert results[-1].kind == "plain"
        assert "生成失败" in results[-1].text and "403" in results[-1].text, results[-1].text
        await plugin.terminate()

    asyncio.run(run())


@case("插件：_build_request 限制张数与尺寸")
def test_build_request(main: types.ModuleType) -> None:
    plugin = new_plugin(main, max_n=3, size="768x768")
    request = plugin._build_request(main.ParsedInput(prompt="猫", n=99))
    assert request.n == 3
    assert request.size == "768x768"

    request = plugin._build_request(main.ParsedInput(prompt="猫"))
    assert request.n == 1

    request = plugin._build_request(main.ParsedInput(prompt="猫", size="800x600"))
    assert request.size == "800x600"


@case("插件：大模型工具 generate_image")
def test_plugin_llm_tool(main: types.ModuleType) -> None:
    async def run() -> None:
        HTTP.reset(lambda call: (200, json.dumps({"data": [{"b64_json": PNG_B64, "revised_prompt": "rp"}]})))
        plugin = new_plugin(main)
        results = [r async for r in plugin.generate_image_tool(FakeEvent(), "一只橘猫")]

        assert results[0].kind == "image"
        assert results[-1].kind == "plain" and "已生成 1 张图片" in results[-1].text
        assert "rp" in results[-1].text
        assert HTTP.calls[0]["json"]["prompt"] == "一只橘猫"

        # 关闭后应当直接拒绝
        HTTP.reset(None)
        plugin2 = new_plugin(main, enable_llm_tool=False)
        blocked = [r async for r in plugin2.generate_image_tool(FakeEvent(), "猫")]
        assert blocked[0].kind == "plain" and "关闭" in blocked[0].text

        # 空提示词
        empty = [r async for r in plugin.generate_image_tool(FakeEvent(), "   ")]
        assert "描述" in empty[0].text

        await plugin.terminate()
        await plugin2.terminate()

    asyncio.run(run())


@case("插件：过期图片清理")
def test_cleanup(main: types.ModuleType) -> None:
    async def run() -> None:
        HTTP.reset(None)
        plugin = new_plugin(main, keep_days=7)
        old = plugin.images_dir / "old.png"
        new = plugin.images_dir / "new.png"
        old.write_bytes(PNG_BYTES)
        new.write_bytes(PNG_BYTES)

        import os
        import time as _time

        past = _time.time() - 30 * 86400
        os.utime(old, (past, past))

        plugin._last_cleanup = 0.0
        plugin._maybe_cleanup()
        assert not old.exists(), "过期图片应被删除"
        assert new.exists(), "新图片不应被删除"

        # keep_days=0 表示永久保留
        plugin2 = new_plugin(main, keep_days=0)
        plugin2._last_cleanup = 0.0
        plugin2._maybe_cleanup()
        await plugin.terminate()
        await plugin2.terminate()

    asyncio.run(run())


# --------------------------------------------------------------------------- #
# 运行器
# --------------------------------------------------------------------------- #
def main() -> int:
    if not MAIN_PY.is_file():
        print(f"[FATAL] 找不到 {MAIN_PY}")
        return 2

    # 沙箱里平台临时目录可能不可写；同时部分环境下 mkdtemp 建出的目录会带
    # 受限 ACL，导致写入被拒，所以这里用普通 mkdir 在插件目录下建临时目录。
    temp_dir = ROOT / "tests" / "_tmp_data"
    import shutil

    shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        module = load_main(temp_dir)
        return _run_cases(module)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _run_cases(module: types.ModuleType) -> int:
    failures: list[str] = []
    for name, func in CASES:
        try:
            func(module)
        except Exception as exc:  # noqa: BLE001 - 测试运行器
            import traceback

            failures.append(name)
            print(f"[FAIL] {name}\n       {type(exc).__name__}: {exc}")
            traceback.print_exc()
        else:
            print(f"[PASS] {name}")

    total = len(CASES)
    print("-" * 60)
    if failures:
        print(f"{len(failures)}/{total} 个用例失败：{', '.join(failures)}")
        return 1
    print(f"{total}/{total} 个用例全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
