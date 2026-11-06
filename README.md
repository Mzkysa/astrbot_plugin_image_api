# astrbot_plugin_image_api

[![CI](https://github.com/Mzkysa/astrbot_plugin_image_api/actions/workflows/ci.yml/badge.svg)](https://github.com/Mzkysa/astrbot_plugin_image_api/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 通过第三方图片生成 API 出图的 AstrBot 插件：一条 `/画图` 指令，或者让大模型自己动手画。

## 功能

- **三类接口开箱即用**
  - `openai`：OpenAI 兼容的 `POST /v1/images/generations`，覆盖 OpenAI、硅基流动、智谱、
    302.AI、NewAPI / one-api 中转站等绝大多数服务商；
  - `sdwebui`：Stable Diffusion WebUI 的 `POST /sdapi/v1/txt2img`，支持负面提示词、
    步数、CFG、采样器、随机种子；
  - `custom`：完全自定义 URL / 请求体 / 取值路径，任何"发个 POST 返回图片链接"的
    三方接口都能接。
- **指令 + 大模型工具**：`/画图` 手动出图；`generate_image` 工具让 LLM 在聊天中自己调用。
- **参数覆盖**：`--size`、`--n`、`--model`、`--neg`、`--seed` 临时覆盖配置。
- **工程细节**：并发上限、单用户冷却、提示词长度限制、网络重试、超时、历史图片自动清理、
  错误信息中文化。

## 安装

1. 把整个 `astrbot_plugin_image_api` 目录放到 AstrBot 的 `data/plugins/` 下；
2. 在 WebUI「插件」页重载插件（或重启 AstrBot）；
3. 打开插件配置，按下方示例填写接口地址、API Key 和模型名。

依赖只有 `aiohttp`（AstrBot 本体已自带），`requirements.txt` 里已声明。

## 配置示例

### OpenAI 官方

| 配置项 | 值 |
| --- | --- |
| `provider` | `openai` |
| `api_base` | `https://api.openai.com/v1` |
| `api_key` | `sk-...` |
| `model` | `gpt-image-1` 或 `dall-e-3` |
| `size` | `1024x1024` |

> `dall-e-3` 只支持 `n=1`，尺寸仅限 `1024x1024` / `1024x1792` / `1792x1024`。

### 硅基流动（Kolors）

| 配置项 | 值 |
| --- | --- |
| `provider` | `openai` |
| `api_base` | `https://api.siliconflow.cn/v1` |
| `api_key` | `sk-...` |
| `model` | `Kwai-Kolors/Kolors` |
| `response_format` | `url` |

### 智谱 CogView

| 配置项 | 值 |
| --- | --- |
| `provider` | `openai` |
| `api_base` | `https://open.bigmodel.cn/api/paas/v4` |
| `api_key` | 智谱开放平台的 API Key |
| `model` | `cogview-4` |
| `size` | `1024x1024` |

### NewAPI / one-api 等中转站

| 配置项 | 值 |
| --- | --- |
| `provider` | `openai` |
| `api_base` | `https://你的中转站/v1` |
| `api_key` | 中转站令牌 |
| `model` | 中转站里配置的图片模型名，如 `flux-schnell` |

### Stable Diffusion WebUI

先用 `./webui.sh --api --listen` 启动，然后：

| 配置项 | 值 |
| --- | --- |
| `provider` | `sdwebui` |
| `sd_base_url` | `http://127.0.0.1:7860` |
| `sd_steps` / `sd_cfg_scale` / `sd_sampler` | 按显卡与喜好调整 |

### 自定义接口

假设接口是：

```http
POST https://api.example.com/v1/gen
{"q": "一只猫", "num": 1, "w": 1024, "h": 1024}
```

返回：

```json
{"result": {"images": [{"url": "https://cdn.example.com/a.png"}]}}
```

则配置：

| 配置项 | 值 |
| --- | --- |
| `provider` | `custom` |
| `custom_url` | `https://api.example.com/v1/gen` |
| `custom_method` | `POST` |
| `custom_body` | `{"q": "{prompt}", "num": {n}, "w": {width}, "h": {height}}` |
| `custom_image_path` | `result.images.0.url` |

占位符说明：`{prompt}`、`{negative_prompt}`、`{size}`、`{width}`、`{height}`、`{n}`、
`{model}`、`{seed}`。在 URL 中会自动做 URL 编码，在 JSON 请求体中会自动做字符串转义。
`custom_image_path` 留空时，插件会自动在响应里找第一个 http 链接或 base64 图片。

> 需要自定义请求头（比如 `x-api-key`）时，把 `auth_mode` 设成 `api-key` 并填写
> `api_key_header`；其他固定请求头可以写进 `extra_headers`。

## 使用

```
/画图 赛博朋克风格的重庆夜景，霓虹灯，雨夜街道
/画图 一只在月球上钓鱼的猫 --size 1024x1792 --n 2
/画图 水墨山水 --model flux-schnell
/画图 帮助
```

`/绘图`、`/文生图`、`/draw`、`/img` 与 `/画图` 等价。

| 参数 | 说明 |
| --- | --- |
| `--size 1024x1024` | 覆盖本次生成的尺寸 |
| `--n 2` | 本次生成张数（不超过 `max_n`） |
| `--model xxx` | 临时指定模型 |
| `--neg "模糊, 低质量"` | 负面提示词（SD WebUI 生效） |
| `--seed 12345` | 固定随机种子（SD WebUI 生效） |

### 大模型工具

配置里开启 `enable_llm_tool` 后，插件会向 AstrBot 注册 `generate_image` 工具。
用户说"帮我画一只橘猫"，大模型即可自行调用，生成的图片会直接发到当前会话。

## 数据存放

生成的图片保存在 `data/plugin_data/astrbot_plugin_image_api/images/`，
超过 `keep_days` 天会自动清理（每小时最多扫描一次）。

## 常见问题

**401 / 403**：API Key 不对，或该 Key 没有图片模型的权限。

**404**：`api_base` 少了 `/v1`，或者模型名写错。

**"接口没有返回任何图片"**：接口返回结构不是标准的 `data[].url` / `data[].b64_json`。
把 `provider` 改成 `custom` 并用 `custom_image_path` 指定真实字段路径。

**生成很慢 / 超时**：把 `timeout` 调大（默认 180 秒），或换更快的模型（如 `flux-schnell`）。

**QQ 官方接口发不出图**：部分平台对图片格式有限制，插件已统一保存为真实图片文件再发送；
若仍失败，请检查协议端配置。

## 自测

`tests/test_offline.py` 用桩模块替换 AstrBot 与 aiohttp，在完全离线的环境里验证
指令解析、三种 provider 的请求构造与响应解析、错误处理、冷却与图片落盘，
无需安装 AstrBot 或 aiohttp：

```bash
python tests/test_offline.py
```

## 许可

MIT
