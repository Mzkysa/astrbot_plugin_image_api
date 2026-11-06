# -*- coding: utf-8 -*-
"""astrbot_plugin_image_api

通过第三方图片生成 API 生成图片的 AstrBot 插件。

支持三类接口：

1. ``openai``：OpenAI 兼容的 ``/v1/images/generations`` 接口。绝大多数中转站、
   云厂商（OpenAI、硅基流动、智谱、302.AI、NewAPI、one-api 等）都能用这一种。
2. ``sdwebui``：本地/远程 Stable Diffusion WebUI 的 ``/sdapi/v1/txt2img`` 接口。
3. ``custom``：完全自定义的 HTTP 接口，通过 ``{prompt}`` 等占位符拼装请求体，
   再按 JSON 路径取出图片。

主要能力：

* ``/画图 <提示词>`` 指令，支持 ``--size`` / ``--n`` / ``--model`` / ``--neg`` /
  ``--seed`` 参数；
* 注册 ``generate_image`` 大模型工具，让 LLM 在聊天中自动调用；
* 生成结果保存到 ``data/plugin_data/astrbot_plugin_image_api/images`` 后发送；
* 并发限制、单用户冷却、提示词长度限制、历史图片自动清理。
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import re
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator
from urllib.parse import quote

import aiohttp

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, StarTools

PLUGIN_NAME = "astrbot_plugin_image_api"
PLUGIN_VERSION = "1.0.0"

# 指令名。CJK 指令允许与提示词连写（如「画图一只猫」），ASCII 指令需要分隔符。
CMD_CJK: tuple[str, ...] = ("画图", "绘图", "文生图")
CMD_ASCII: tuple[str, ...] = ("draw", "img")

_COMMAND_RE = re.compile(
    r"^\s*[/／!#!！.。]?\s*(?:"
    + "|".join(re.escape(name) for name in CMD_CJK)
    + r"|(?:"
    + "|".join(re.escape(name) for name in CMD_ASCII)
    + r")(?=$|[\s:：,，]))\s*",
    re.IGNORECASE,
)

_OPTION_RE = re.compile(
    r"(?<!\S)--(?P<key>[A-Za-z][\w-]*)"
    r"(?:=(?P<eq>\S+)|\s+(?P<val>\"[^\"]*\"|'[^']*'|\S+))?"
)

_OPTION_ALIASES: dict[str, str] = {
    "size": "size",
    "尺寸": "size",
    "n": "n",
    "num": "n",
    "数量": "n",
    "model": "model",
    "模型": "model",
    "neg": "negative_prompt",
    "negative": "negative_prompt",
    "负面": "negative_prompt",
    "seed": "seed",
    "种子": "seed",
}

HELP_TEXT = (
    "🎨 图片生成插件使用说明\n"
    "\n"
    "• /画图 <提示词> —— 生成图片（/绘图、/文生图、/draw 等价）\n"
    "\n"
    "可选参数：\n"
    "• --size 1024x1024  指定尺寸\n"
    "• --n 2             一次生成多张\n"
    "• --model xxx       临时指定模型\n"
    "• --neg \"模糊, 低质量\"  负面提示词（SD WebUI 生效）\n"
    "• --seed 12345      固定随机种子（SD WebUI 生效）\n"
    "\n"
    "示例：\n"
    "/画图 赛博朋克风格的重庆夜景，霓虹灯，雨夜街道 --size 1024x1024"
)

DEFAULT_SD_NEGATIVE = (
    "lowres, bad anatomy, bad hands, worst quality, low quality, "
    "jpeg artifacts, watermark, signature"
)

MAX_DOWNLOAD_BYTES = 32 * 1024 * 1024  # 单张图片最大 32MB


# --------------------------------------------------------------------------- #
# 通用小工具
# --------------------------------------------------------------------------- #
def as_int(value: Any, default: int = 0) -> int:
    """尽量把配置值转成 int，失败时返回默认值。"""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def as_float(value: Any, default: float = 0.0) -> float:
    """尽量把配置值转成 float，失败时返回默认值。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_bool(value: Any, default: bool = False) -> bool:
    """尽量把配置值转成 bool，兼容 "true"/"1"/"on" 之类的字符串。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "y"}
    return default


def detect_image_ext(data: bytes) -> str:
    """根据文件头猜测图片扩展名。"""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return ".gif"
    return ".png"


def parse_size(size: str) -> tuple[int, int] | None:
    """把 ``1024x768`` 解析成 ``(1024, 768)``。"""
    match = re.fullmatch(r"\s*(\d{2,5})\s*[xX*×]\s*(\d{2,5})\s*", size or "")
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def normalize_size(size: str) -> str:
    """把各种写法的尺寸统一成 ``宽x高``，非法时返回空串。"""
    parsed = parse_size(size)
    if not parsed:
        return ""
    return f"{parsed[0]}x{parsed[1]}"


# --------------------------------------------------------------------------- #
# 指令文本解析
# --------------------------------------------------------------------------- #
@dataclass
class ParsedInput:
    """一条 /画图 指令解析后的结果。"""

    prompt: str = ""
    size: str = ""
    n: int = 0
    model: str = ""
    negative_prompt: str = ""
    seed: int | None = None
    help_requested: bool = False


def strip_command(text: str) -> str:
    """去掉开头的指令名（含可能的 / 前缀）。"""
    return _COMMAND_RE.sub("", (text or "").strip(), count=1).strip()


def parse_command_text(raw: str) -> ParsedInput:
    """解析 ``/画图 xxx --size 512x512`` 这类文本。

    兼容两种情况：``raw`` 保留完整指令（``/画图 一只猫``），或框架已经把指令名
    剥离（``一只猫``）。
    """
    text = strip_command(raw)
    parsed = ParsedInput()

    if not text or text.lower() in {"help", "-h", "--help", "帮助", "菜单", "用法"}:
        parsed.help_requested = True
        return parsed

    spans: list[tuple[int, int]] = []
    for match in _OPTION_RE.finditer(text):
        key = _OPTION_ALIASES.get(match.group("key").lower())
        if key is None:
            continue
        value = (match.group("eq") or match.group("val") or "").strip().strip("\"'")
        spans.append((match.start(), match.end()))

        if key == "size":
            parsed.size = normalize_size(value)
        elif key == "n":
            parsed.n = max(0, min(10, as_int(value, 0)))
        elif key == "model":
            parsed.model = value
        elif key == "negative_prompt":
            parsed.negative_prompt = value
        elif key == "seed":
            try:
                parsed.seed = int(value)
            except ValueError:
                parsed.seed = None

    for start, end in reversed(spans):
        text = f"{text[:start]} {text[end:]}"

    parsed.prompt = " ".join(text.split())
    if not parsed.prompt:
        parsed.help_requested = True
    return parsed


# --------------------------------------------------------------------------- #
# 异常与数据结构
# --------------------------------------------------------------------------- #
class ImageGenError(Exception):
    """生成失败。异常信息会直接展示给用户，因此要写成友好的中文。"""


@dataclass
class ImageItem:
    """接口返回的一张图片（base64 或 URL 二选一）。"""

    b64: str = ""
    url: str = ""
    revised_prompt: str = ""


@dataclass
class GenRequest:
    """一次生成请求。"""

    prompt: str
    size: str = "1024x1024"
    n: int = 1
    model: str = ""
    negative_prompt: str = ""
    seed: int | None = None


def format_api_error(err: Any) -> str:
    """把各家五花八门的错误结构压成一句话。"""
    if isinstance(err, dict):
        message = err.get("message") or err.get("msg") or err.get("error")
        if message:
            return str(message)[:300]
        return json.dumps(err, ensure_ascii=False)[:300]
    return str(err)[:300]


def format_http_error(status: int, text: str) -> str:
    """生成友好的 HTTP 错误提示。"""
    detail = (text or "").strip()
    try:
        payload = json.loads(detail)
    except (json.JSONDecodeError, ValueError):
        payload = None
    if isinstance(payload, dict):
        err = payload.get("error") or payload
        detail = format_api_error(err)

    hints = {
        400: "请求参数被接口拒绝，请检查模型名称、尺寸是否被该模型支持",
        401: "API Key 无效或未授权",
        403: "没有访问该模型的权限",
        404: "接口地址或模型不存在，注意 Base URL 是否包含 /v1",
        429: "请求过于频繁或额度已用尽",
        500: "接口服务端错误",
        502: "接口网关错误",
        503: "接口服务暂时不可用",
    }
    message = f"接口返回 HTTP {status}"
    hint = hints.get(status, "")
    if hint:
        message += f"（{hint}）"
    if detail:
        message += f"：{detail}"
    return message


def replace_placeholders(text: str, values: dict[str, str], *, url_mode: bool) -> str:
    """替换 ``{prompt}`` 这类占位符。

    ``url_mode=True`` 时做 URL 编码；否则做 JSON 字符串转义（用于拼 JSON 请求体）。
    """
    pattern = re.compile(r"\{([a-z_]+)\}")

    def _sub(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in values:
            return match.group(0)
        value = str(values[key])
        if url_mode:
            return quote(value, safe="")
        return json.dumps(value, ensure_ascii=False)[1:-1]

    return pattern.sub(_sub, text)


def json_path_get(payload: Any, path: str) -> Any:
    """按 ``data.0.url`` 这样的路径取值，取不到返回 None。"""
    if not path:
        return None
    current = payload
    for part in path.split("."):
        if not part:
            continue
        if isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                return None
        elif isinstance(current, dict):
            current = current.get(part)
        else:
            return None
        if current is None:
            return None
    return current


def auto_find_image(payload: Any) -> ImageItem | None:
    """在任意结构的 JSON 中找第一个看起来像图片的字段。"""
    stack: list[Any] = [payload]
    b64_candidate = ""
    while stack:
        current = stack.pop(0)
        if isinstance(current, dict):
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)
        elif isinstance(current, str):
            value = current.strip()
            if value.startswith(("http://", "https://")):
                return ImageItem(url=value)
            if value.startswith("data:image/"):
                return ImageItem(b64=value)
            if len(value) > 512 and not b64_candidate:
                b64_candidate = value
    if b64_candidate:
        return ImageItem(b64=b64_candidate)
    return None


# --------------------------------------------------------------------------- #
# 图片接口客户端
# --------------------------------------------------------------------------- #
class ImageAPIClient:
    """对三类第三方图片接口的统一封装。"""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config or {}
        self._session: aiohttp.ClientSession | None = None
        concurrency = max(1, as_int(self._cfg("max_concurrency", 2), 2))
        self._semaphore = asyncio.Semaphore(concurrency)

    # -- 配置读取 ---------------------------------------------------------- #
    def _cfg(self, key: str, default: Any = None) -> Any:
        value = self.config.get(key, default)
        return default if value is None else value

    # -- 会话管理 ---------------------------------------------------------- #
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=float(as_int(self._cfg("timeout", 180), 180)))
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        """插件卸载/停用时关闭连接池。"""
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    # -- 请求头 ------------------------------------------------------------ #
    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        api_key = str(self._cfg("api_key", "") or "").strip()
        auth_mode = str(self._cfg("auth_mode", "bearer") or "bearer").strip().lower()
        if api_key and auth_mode != "none":
            if auth_mode == "api-key":
                header_name = str(self._cfg("api_key_header", "X-API-Key") or "X-API-Key")
                headers[header_name.strip()] = api_key
            else:
                headers["Authorization"] = f"Bearer {api_key}"
        extra = self._cfg("extra_headers", {})
        if isinstance(extra, dict):
            for key, value in extra.items():
                if key and value not in (None, ""):
                    headers[str(key)] = str(value)
        return headers

    # -- 底层请求 ---------------------------------------------------------- #
    async def _request_json(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any] | None,
    ) -> dict[str, Any]:
        session = await self._get_session()
        attempts = max(1, as_int(self._cfg("retry", 1), 1) + 1)
        last_error: Exception | None = None

        for attempt in range(attempts):
            try:
                kwargs: dict[str, Any] = {"headers": headers}
                if method.upper() == "GET":
                    kwargs["params"] = {k: str(v) for k, v in (payload or {}).items()}
                else:
                    kwargs["json"] = payload or {}
                async with session.request(method.upper(), url, **kwargs) as resp:
                    text = await resp.text()
                    if resp.status >= 400:
                        raise ImageGenError(format_http_error(resp.status, text))
                    try:
                        data = json.loads(text)
                    except (json.JSONDecodeError, ValueError) as exc:
                        raise ImageGenError(f"接口返回的内容不是合法 JSON：{text[:200]}") from exc
                    if isinstance(data, dict) and data.get("error"):
                        raise ImageGenError(format_api_error(data["error"]))
                    if not isinstance(data, dict):
                        raise ImageGenError("接口返回的 JSON 不是对象，无法解析。")
                    return data
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
                last_error = exc
                if attempt + 1 >= attempts:
                    break
                await asyncio.sleep(float(attempt + 1))

        raise ImageGenError(f"网络请求失败：{last_error}")

    async def _download(self, url: str) -> bytes:
        session = await self._get_session()
        try:
            async with session.get(url) as resp:
                if resp.status >= 400:
                    raise ImageGenError(f"下载生成结果失败：HTTP {resp.status}")
                length = resp.content_length
                if length and length > MAX_DOWNLOAD_BYTES:
                    raise ImageGenError("生成的图片体积过大，已取消下载。")
                data = await resp.content.read(MAX_DOWNLOAD_BYTES + 1)
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
            raise ImageGenError(f"下载生成结果失败：{exc}") from exc

        if len(data) > MAX_DOWNLOAD_BYTES:
            raise ImageGenError("生成的图片体积过大，已取消下载。")
        if not data:
            raise ImageGenError("下载到的图片内容为空。")
        return data

    @staticmethod
    def _decode_b64(raw: str) -> bytes:
        value = (raw or "").strip()
        if value.startswith("data:"):
            _, _, value = value.partition(",")
        value = "".join(value.split())
        padding = (-len(value)) % 4
        try:
            return base64.b64decode(value + "=" * padding)
        except (binascii.Error, ValueError) as exc:
            raise ImageGenError("接口返回的 base64 图片数据无法解码。") from exc

    # -- 对外入口 ---------------------------------------------------------- #
    async def generate(self, request: GenRequest) -> list[tuple[bytes, str]]:
        """执行一次生成，返回 ``[(图片字节, 改写后的提示词), ...]``。"""
        provider = str(self._cfg("provider", "openai") or "openai").strip().lower()

        async with self._semaphore:
            if provider == "sdwebui":
                items = await self._generate_sdwebui(request)
            elif provider == "custom":
                items = await self._generate_custom(request)
            else:
                items = await self._generate_openai(request)

        if not items:
            raise ImageGenError("接口没有返回任何图片，请检查模型名称与参数配置。")

        results: list[tuple[bytes, str]] = []
        for item in items:
            if item.b64:
                results.append((self._decode_b64(item.b64), item.revised_prompt))
            elif item.url:
                results.append((await self._download(item.url), item.revised_prompt))
        if not results:
            raise ImageGenError("接口返回的图片数据为空。")
        return results

    # -- OpenAI 兼容 ------------------------------------------------------- #
    def _openai_url(self) -> str:
        base = str(self._cfg("api_base", "https://api.openai.com/v1") or "").strip().rstrip("/")
        if not base:
            raise ImageGenError("尚未配置 api_base（接口地址），请在插件配置中填写。")
        if base.endswith("/images/generations"):
            return base
        return f"{base}/images/generations"

    def _build_openai_body(self, request: GenRequest) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": request.model or str(self._cfg("model", "gpt-image-1") or ""),
            "prompt": request.prompt,
            "n": max(1, request.n),
        }
        if request.size:
            body["size"] = request.size

        response_format = str(self._cfg("response_format", "") or "").strip()
        if response_format in {"url", "b64_json"}:
            body["response_format"] = response_format

        quality = str(self._cfg("quality", "") or "").strip()
        if quality:
            body["quality"] = quality

        extra = self._cfg("extra_body", {})
        if isinstance(extra, dict) and extra:
            body.update(extra)
        return body

    @staticmethod
    def _extract_openai_items(payload: dict[str, Any]) -> list[ImageItem]:
        items: list[ImageItem] = []
        data = payload.get("data")
        if isinstance(data, dict):
            data = [data]
        if isinstance(data, list):
            for entry in data:
                if not isinstance(entry, dict):
                    continue
                revised = str(entry.get("revised_prompt") or "")
                b64 = entry.get("b64_json") or entry.get("base64")
                url = entry.get("url") or entry.get("image_url")
                if isinstance(b64, str) and b64:
                    items.append(ImageItem(b64=b64, revised_prompt=revised))
                elif isinstance(url, str) and url:
                    items.append(ImageItem(url=url, revised_prompt=revised))
        return items

    async def _generate_openai(self, request: GenRequest) -> list[ImageItem]:
        url = self._openai_url()
        payload = await self._request_json(
            "POST", url, self._headers(), self._build_openai_body(request)
        )
        items = self._extract_openai_items(payload)
        if not items:
            found = auto_find_image(payload)
            if found:
                items.append(found)
        return items

    # -- Stable Diffusion WebUI -------------------------------------------- #
    async def _generate_sdwebui(self, request: GenRequest) -> list[ImageItem]:
        base = str(self._cfg("sd_base_url", "http://127.0.0.1:7860") or "").strip().rstrip("/")
        if not base:
            raise ImageGenError("尚未配置 sd_base_url（SD WebUI 地址）。")
        url = base if base.endswith("/sdapi/v1/txt2img") else f"{base}/sdapi/v1/txt2img"

        width, height = parse_size(request.size) or (512, 768)
        payload: dict[str, Any] = {
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt
            or str(self._cfg("sd_negative_prompt", DEFAULT_SD_NEGATIVE) or ""),
            "steps": as_int(self._cfg("sd_steps", 25), 25),
            "cfg_scale": as_float(self._cfg("sd_cfg_scale", 7.0), 7.0),
            "width": width,
            "height": height,
            "batch_size": max(1, min(request.n, 4)),
            "sampler_name": str(self._cfg("sd_sampler", "DPM++ 2M Karras") or ""),
        }
        if request.seed is not None:
            payload["seed"] = request.seed

        data = await self._request_json("POST", url, self._headers(), payload)
        images = data.get("images")
        items: list[ImageItem] = []
        if isinstance(images, list):
            for entry in images:
                if isinstance(entry, str) and entry.strip():
                    items.append(ImageItem(b64=entry))
        return items

    # -- 自定义接口 -------------------------------------------------------- #
    async def _generate_custom(self, request: GenRequest) -> list[ImageItem]:
        url_template = str(self._cfg("custom_url", "") or "").strip()
        if not url_template:
            raise ImageGenError("尚未配置 custom_url（自定义接口地址）。")

        width, height = parse_size(request.size) or (0, 0)
        values = {
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "size": request.size,
            "width": str(width or ""),
            "height": str(height or ""),
            "n": str(request.n),
            "model": request.model or str(self._cfg("model", "") or ""),
            "seed": "" if request.seed is None else str(request.seed),
        }

        url = replace_placeholders(url_template, values, url_mode=True)
        template = str(self._cfg("custom_body", "") or "").strip() or '{"prompt": "{prompt}"}'
        body_text = replace_placeholders(template, values, url_mode=False)
        try:
            payload = json.loads(body_text)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ImageGenError(f"custom_body 不是合法 JSON（替换占位符后）：{exc}") from exc
        if not isinstance(payload, dict):
            raise ImageGenError("custom_body 替换后必须是一个 JSON 对象。")

        method = str(self._cfg("custom_method", "POST") or "POST").upper()
        data = await self._request_json(method, url, self._headers(), payload)

        path = str(self._cfg("custom_image_path", "") or "").strip()
        item: ImageItem | None = None
        if path:
            value = json_path_get(data, path)
            if isinstance(value, str) and value:
                if value.startswith(("http://", "https://")):
                    item = ImageItem(url=value)
                else:
                    item = ImageItem(b64=value)
            elif isinstance(value, list) and value and isinstance(value[0], str):
                first = value[0]
                item = ImageItem(url=first) if first.startswith("http") else ImageItem(b64=first)
        if item is None:
            item = auto_find_image(data)
        return [item] if item else []


# --------------------------------------------------------------------------- #
# 插件主体
# --------------------------------------------------------------------------- #
class ImageAPIPlugin(Star):
    """第三方 API 图片生成插件。"""

    def __init__(self, context: Context, config: AstrBotConfig | None = None) -> None:
        super().__init__(context)
        self.config: dict[str, Any] = dict(config or {})
        self.client = ImageAPIClient(self.config)
        self.data_dir = self._resolve_data_dir()
        self.images_dir = self.data_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)

        self._cooldowns: dict[str, float] = {}
        self._last_cleanup = 0.0

    # -- 基础工具 ---------------------------------------------------------- #
    @staticmethod
    def _resolve_data_dir() -> Path:
        """优先使用 AstrBot 的 data/plugin_data 目录，失败时退回插件目录。"""
        try:
            return Path(StarTools.get_data_dir(PLUGIN_NAME))
        except Exception:  # noqa: BLE001 - 兼容不同 AstrBot 版本
            return Path(__file__).resolve().parent / "data"

    def _cfg(self, key: str, default: Any = None) -> Any:
        value = self.config.get(key, default)
        return default if value is None else value

    def _build_request(self, parsed: ParsedInput) -> GenRequest:
        size = parsed.size or normalize_size(str(self._cfg("size", "1024x1024") or ""))
        if not size:
            size = "1024x1024"

        default_n = max(1, as_int(self._cfg("default_n", 1), 1))
        max_n = max(1, as_int(self._cfg("max_n", 4), 4))
        n = min(max(1, parsed.n or default_n), max_n)

        model = parsed.model or str(self._cfg("model", "") or "")
        return GenRequest(
            prompt=parsed.prompt,
            size=size,
            n=n,
            model=model,
            negative_prompt=parsed.negative_prompt,
            seed=parsed.seed,
        )

    def _check_cooldown(self, user_id: str) -> float:
        """返回还需要等待的秒数，0 表示可以继续。"""
        cooldown = as_int(self._cfg("cooldown_seconds", 10), 10)
        if cooldown <= 0 or not user_id:
            return 0.0
        now = time.monotonic()
        last = self._cooldowns.get(user_id, 0.0)
        elapsed = now - last
        if elapsed < cooldown:
            return cooldown - elapsed
        return 0.0

    async def _save_image(self, data: bytes) -> Path:
        """把图片写到本地并返回路径。"""
        filename = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}{detect_image_ext(data)}"
        path = self.images_dir / filename
        await asyncio.to_thread(path.write_bytes, data)
        return path

    def _maybe_cleanup(self) -> None:
        """按 keep_days 清理历史图片，每小时最多执行一次。"""
        now = time.time()
        if now - self._last_cleanup < 3600:
            return
        self._last_cleanup = now

        keep_days = as_int(self._cfg("keep_days", 7), 7)
        if keep_days <= 0:
            return
        cutoff = now - keep_days * 86400
        removed = 0
        try:
            for file in self.images_dir.iterdir():
                if file.is_file() and file.stat().st_mtime < cutoff:
                    file.unlink(missing_ok=True)
                    removed += 1
        except OSError as exc:  # 清理失败不影响主流程
            logger.warning(f"[image_api] 清理历史图片失败：{exc}")
            return
        if removed:
            logger.info(f"[image_api] 已清理 {removed} 张过期图片。")

    async def _generate_and_save(self, request: GenRequest) -> tuple[list[Path], str]:
        """调用接口生成图片并落盘，返回 (路径列表, 被改写的提示词)。"""
        results = await self.client.generate(request)
        paths: list[Path] = []
        revised = ""
        for data, revised_prompt in results:
            paths.append(await self._save_image(data))
            if revised_prompt and not revised:
                revised = revised_prompt
        self._maybe_cleanup()
        logger.info(f"[image_api] 生成成功，共 {len(paths)} 张：{request.prompt[:60]}")
        return paths, revised

    # -- 指令 -------------------------------------------------------------- #
    @filter.command("画图")
    async def draw_cmd(self, event: AstrMessageEvent) -> AsyncIterator[Any]:
        """根据提示词生成图片。"""
        async for result in self._handle_draw(event):
            yield result

    @filter.command("绘图")
    async def draw_cmd_alias_1(self, event: AstrMessageEvent) -> AsyncIterator[Any]:
        """根据提示词生成图片（画图 的同义词）。"""
        async for result in self._handle_draw(event):
            yield result

    @filter.command("文生图")
    async def draw_cmd_alias_2(self, event: AstrMessageEvent) -> AsyncIterator[Any]:
        """根据提示词生成图片（画图 的同义词）。"""
        async for result in self._handle_draw(event):
            yield result

    @filter.command("draw")
    async def draw_cmd_alias_3(self, event: AstrMessageEvent) -> AsyncIterator[Any]:
        """Generate an image from a text prompt."""
        async for result in self._handle_draw(event):
            yield result

    async def _handle_draw(self, event: AstrMessageEvent) -> AsyncIterator[Any]:
        parsed = parse_command_text(event.message_str or "")

        if parsed.help_requested:
            yield event.plain_result(HELP_TEXT)
            return

        max_length = max(10, as_int(self._cfg("max_prompt_length", 800), 800))
        if len(parsed.prompt) > max_length:
            yield event.plain_result(
                f"提示词太长了（{len(parsed.prompt)} 字），请控制在 {max_length} 字以内。"
            )
            return

        user_id = str(event.get_sender_id() or "")
        wait = self._check_cooldown(user_id)
        if wait > 0:
            yield event.plain_result(f"⏳ 请求太频繁啦，请等待 {wait:.0f} 秒后再试。")
            return

        request = self._build_request(parsed)
        if as_bool(self._cfg("notify_before_generate", True), True):
            yield event.plain_result("🎨 正在生成图片，请稍候…")

        self._cooldowns[user_id] = time.monotonic()
        started = time.monotonic()
        try:
            paths, revised = await self._generate_and_save(request)
        except ImageGenError as exc:
            logger.warning(f"[image_api] 生成失败：{exc}")
            yield event.plain_result(f"❌ 生成失败：{exc}")
            return
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - 兜底，避免插件崩溃
            logger.error(f"[image_api] 生成图片时出现未预期的错误：{exc}\n{traceback.format_exc()}")
            yield event.plain_result("❌ 生成失败：插件内部错误，详情请查看 AstrBot 日志。")
            return

        for path in paths:
            yield event.image_result(str(path.resolve()))

        elapsed = time.monotonic() - started
        summary = f"✅ 已生成 {len(paths)} 张图片，耗时 {elapsed:.1f} 秒。"
        if revised:
            summary += f"\n模型改写后的提示词：{revised}"
        yield event.plain_result(summary)

    # -- 大模型工具 -------------------------------------------------------- #
    @filter.llm_tool(name="generate_image")
    async def generate_image_tool(self, event: AstrMessageEvent, prompt: str):
        """根据文字描述生成一张图片，并把图片发送给用户。

        Args:
            prompt(string): 想要生成的图片的详细描述，尽量包含主体、风格、构图、光线等
        """
        if not as_bool(self._cfg("enable_llm_tool", True), True):
            yield event.plain_result("图片生成功能当前已在插件配置中关闭。")
            return

        text = (prompt or "").strip()
        if not text:
            yield event.plain_result("请先描述你想生成的画面内容。")
            return

        max_length = max(10, as_int(self._cfg("max_prompt_length", 800), 800))
        if len(text) > max_length:
            yield event.plain_result(f"提示词太长了，请控制在 {max_length} 字以内。")
            return

        parsed = ParsedInput(prompt=text)
        request = self._build_request(parsed)

        try:
            paths, revised = await self._generate_and_save(request)
        except ImageGenError as exc:
            logger.warning(f"[image_api] 工具调用生成失败：{exc}")
            yield event.plain_result(f"图片生成失败：{exc}")
            return
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[image_api] 工具调用出现未预期的错误：{exc}\n{traceback.format_exc()}")
            yield event.plain_result("图片生成失败：插件内部错误。")
            return

        for path in paths:
            yield event.image_result(str(path.resolve()))

        message = f"已生成 {len(paths)} 张图片并发送给用户。"
        if revised:
            message += f"（模型改写后的提示词：{revised}）"
        yield event.plain_result(message)

    # -- 生命周期 ---------------------------------------------------------- #
    async def terminate(self) -> None:
        """插件被停用/卸载时释放 HTTP 连接。"""
        await self.client.close()
