# B05 交付报告：标准工具调用 + 工具智能体（Tool Agent）

## 功能

### 4 个标准工具（[tools.py](../tools.py)）

| 工具 | 用途 | 参数 | source |
|---|---|---|---|
| `get_current_environment()` | 当前环境状态（温度/湿度/光照） | 无 | SIMULATED |
| `get_environment_history(hours=24)` | 最近 N 小时历史数据（SQLite，每 30 分钟一个点） | `hours` 为正数 | SIMULATED |
| `get_weather_forecast(location)` | 指定地点未来 3 天逐小时天气（Open-Meteo） | `location` 必填，目前仅支持「南京」 | PUBLIC |
| `search_tomato_knowledge(query)` | 番茄知识库检索最相关 2 个段落（TF-IDF） | `query` 必填 | KNOWLEDGE_BASE |

### 统一 5 字段返回

所有工具函数返回同一个字典，任何异常都转成该结构返回，绝不向外抛异常（`python tools.py` 可独立运行演示）：

| 字段 | 说明 |
|---|---|
| `status` | 只能是 `ok` / `unavailable`（网络超时、外部服务失败）/ `error`（本地错误） |
| `source` | `SIMULATED` / `PUBLIC` / `KNOWLEDGE_BASE` |
| `time` | 工具执行时间 |
| `data` | 业务数据（失败时为 `null`） |
| `error` | 错误说明（无错误时为 `null`） |

边界情况验证：`hours=0`、`hours="abc"`、不支持的 `location`、空 `query` 均返回 `status=error` 的字典而非抛异常；注入 `WeatherError` → `unavailable`，注入其他异常 → `error`。

### 工具智能体（[agent.py](../agent.py)）

- `TOOL_DEFINITIONS`：4 个工具的 OpenAI function calling 格式定义（name / description / parameters JSON Schema）。查天气必须提供 `location`：JSON Schema 中 `required`，description 要求模型在用户未说明地点时先询问、不得猜测。
- `TOOL_DISPATCH`：工具名 → tools.py 真实函数。
- `run_agent()` 闭环：问题 + tools 发给 LLM → 收到 `tool_calls` 用 `json.loads` 解析函数名与参数 → Python 端真实执行 → 结果打包为 `role: tool` 消息回传 → LLM 生成最终回答。
- **防死循环**：`MAX_TOOL_CALLS = 3`，最多执行 3 次工具；超出后拒绝执行并告知，最后强制一轮不带 tools 的收尾请求。
- **容错**：参数 JSON 解析失败、未知工具、参数不匹配、工具内部异常，全部转成统一 error 结果回给模型，闭环不崩溃。
- 终端日志：每次调用打印 `[Tool Call] 调用了 X，参数：{...}` 与 `[Tool Result] 状态：ok/unavailable/error`。

### 接口与页面

- `POST /agent/chat`：body `{"question": "..."}`，返回 `{question, answer, tool_calls}`；空问题 400，LLM 失败 502（错误分类诊断沿用 /ask 的 `_llm_error_response`）。
- 页面顶部新增「AI 助手主控台」：输入框（支持回车）+ 发送按钮，对话区展示用户问题与 AI 回答卡片（含「工具调用 N 次」元信息），失败显示红色错误卡片。
- 原 /ask 同步升级为工具闭环（空 POST 沿用固定问题，A/B 测试不受影响）；状态数据抽到 [state.py](../state.py)，供 main.py 与 tools.py 共用，避免循环导入。

## 测试结果（真实 LLM 调用）

### 多工具混合调用成功

| 问题 | 实际调用 | 结果 |
|---|---|---|
| 「当前环境怎么样？」 | `get_current_environment` | 回答引用 SIMULATED 数据 ✓ |
| 「帮我查一下最近 24 小时的历史环境数据」 | `get_environment_history(hours=24)` | 43 个采样点，正确总结趋势 ✓ |
| 「南京明天会下雨吗？」 | `get_weather_forecast(location=南京)` | 逐小时降水概率，PUBLIC 来源 ✓ |
| 「番茄开花期适宜多少温度？」 | `search_tomato_knowledge` | 带来源引用回答 ✓ |
| 「未来天气怎么样？」（未说明地点） | 0 次调用 | 模型反问地点，未猜测 ✓ |

四个工具均被 LLM 自动、正确路由；终端同步打印对应 `[Tool Call]` / `[Tool Result]` 日志。

### 断网失败降级

- 断网时 LLM 请求失败：/ask 与 /agent/chat 返回 502 `{"error": "API调用失败：网络或密钥错误"}`，页面显示红色错误卡片，**不伪造任何数据**，终端打印分类诊断（密钥错误 vs 网络问题）+ 完整 traceback 写入 app.log。
- 工具层网络异常（如天气接口超时）：工具返回 `status=unavailable` 并附原因，LLM 如实告知「数据暂时不可用」，不编造结果。
- 断网期间页面上不残留、不显示旧数据。

### 防死循环与容错（单元测试）

- 模拟 LLM 每轮都返回 tool_calls：恰好执行 3 次后强制收尾，返回最终回答 ✓
- 一轮返回多个 tool_calls：全部执行；超限部分拒绝执行并回传 error ✓
- 参数非 JSON / 类型错误 / 未知工具：均转 error 结果回传，闭环不崩溃 ✓

## 文件清单

- 新增：`tools.py`（4 个工具）、`state.py`（状态抽离）、`agent.py`（工具定义 + 闭环）、`docs/B05_Report.md`
- 修改：`main.py`（state 模块化、/ask 接入闭环、新增 /agent/chat）、`llm_client.py`（通用 `chat(messages, tools)` 入口）、`templates/index.html`（AI 助手主控台）、`README.md`
