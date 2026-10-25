# B06 交付报告：传统机器学习闭环 + YOLO 视觉工具

## 功能

### 1. ML 数据管线（[ml_pipeline.py](../ml_pipeline.py)）

- **数据**：SQLite `environment_samples` 全部 336 条（B02 模拟数据，30 分钟间隔），按时间排序。
- **特征**：滑动窗口取最近 6 个点（2.5 小时），提取 8 个特征：`temp_current / temp_mean / temp_delta / temp_std / humidity_current / humidity_delta / light_current / light_mean`。特征只允许使用过去数据；表内没有 `sim_scenario` 列，字段按白名单提取，未来数据与场景字段都不可能混入。
- **标签**：看未来 4 个点（2 小时），任一时刻温度偏离当前温度 >= 3.5°C → `next_window_attention=WATCH`，否则 NORMAL。
  - 阈值决策有数据依据：5 个扰动事件（9~13°C）随机全部落在前 5 天，后 2 天只有纯昼夜变化（2 小时最大偏离 3.96°C）；阈值 >= 4.0 时测试集 WATCH=0（无法评估召回），3.5 时训练/测试都有正样本，故取 3.5（可用 `--watch-threshold` 覆盖）。
- **切分**：前 5 天训练（<= 2026-10-17 21:00）、后 2 天测试；整个窗口跨边界的一律丢弃（9 个）。实测验证：训练集标签末点最大 = 边界、测试集特征首点最小 = 21:30，**严格无时间泄漏**。
- 输出 `ml_data/train.csv`（232 行）+ `ml_data/test.csv`（86 行）。

### 2. 训练、评估与保存（[train_model.py](../train_model.py)）

- 随机森林 `RandomForestClassifier(n_estimators=200, random_state=42)`。
- 保存 `models/attention_model.pkl`（模型 + 特征列表 + 参数 + `model_version: rf_v1` + 训练时间）与 `models/features.json`。

### 3. 推理（[predict_model.py](../predict_model.py)）

- `load_model(path)` 加载 pkl；`predict_attention(model_dict, rows)` 输入最近 6 条数据 → 输出 `{"predicted_attention", "model_version"}`。
- 特征计算与训练完全一致（按 `FEATURES` 顺序取值）；加载时校验 pkl 特征列表与当前代码一致。
- 命令行取 SQLite 最新 6 条演示（[db.py](../db.py) 新增 `latest_samples(n)`）。

### 4. YOLO 视觉工具（[vision_tool.py](../vision_tool.py)）

- `analyze_camera_image(image_path)`：`yolo11n.pt`（本地无则自动下载）推理 → 按类别聚合统计（count + 平均置信度）→ 生成描述（是否有遮挡物：person/car/chair 等 10 类；是否检测到植物：potted plant 等）。
- 统一 5 字段返回：`{"status", "source": "YOLO_CAMERA", "time", "data": {"objects", "description"}, "error"}`；图片不存在 → `error`，模型加载失败 → `unavailable`，绝不抛异常。
- 已加入 Tool Agent 工具箱（[agent.py](../agent.py)）：第 5 个工具定义 + `TOOL_DISPATCH` + SYSTEM_PROMPT 路由规则（问「画面里有什么/有没有遮挡」→ 调用视觉工具）；`image_path` 可选，默认 `static/uploads/test_image.jpg`。

## 测试结果

### ML 模型（测试集 86 行，WATCH=13）

| 模型 | accuracy | WATCH 召回 | WATCH F1 |
|---|---|---|---|
| 朴素基线（全猜 NORMAL） | 0.849 | 0.000 | 0.000 |
| 随机森林 | **0.895** | **0.615** | **0.640** |

混淆矩阵（行=真实，列=预测）：NORMAL 正确 69 / 误报 4；WATCH 正确 8 / 漏报 5。模型相对基线 accuracy +4.6 个百分点，WATCH 召回从 0 提升到 0.615，明显优于基线。

推理演示（SQLite 最新窗口 2026-10-19 18:00~20:30，傍晚降温段）：`predicted_attention = NORMAL`，符合预期（夜间缓降不算急剧变化）。

### 视觉工具（YOLO yolo11n）

| 测试 | 结果 |
|---|---|
| `python vision_tool.py` 独立运行 | 检出 person x4、bus x1，描述含遮挡判断 ✓ |
| 图片不存在 | `status=error` + 「图片不存在」✓ |
| 5 字段 / data 结构 | 严格符合规范 ✓ |
| 工具定义校验 | 5 个工具全部对应可调用函数 ✓ |

### 主控台闭环（真实 LLM 调用）

| 问题 | 表现 |
|---|---|
| 「画面里有什么？」 | 自动调 `analyze_camera_image`，回答 5 个物体 + 「画面中没有检测到植物」——正确指出**场景矛盾**（温室画面无植物）✓ |
| 「摄像头画面有没有遮挡？」 | 自动调视觉工具，回答「存在遮挡风险：4 人和 1 辆车」✓ |
| 图片删除后提问 | 明确报错，不伪造结果 ✓ |

终端日志格式：`[Tool Call] 调用了 analyze_camera_image，参数：{}` / `[Tool Result] 状态：ok`。

## 文件清单

- 新增：`ml_pipeline.py`、`train_model.py`、`predict_model.py`、`vision_tool.py`、`docs/B06_Report.md`、`static/uploads/`（测试图，gitignore）
- 修改：`db.py`（`latest_samples`）、`agent.py`（第 5 个工具）、`requirements.txt`（scikit-learn、ultralytics）、`.gitignore`（`ml_data/`、`models/`、`*.pt`、`static/uploads/*`）
- 生成本地产物（不入库，可随时重生成）：`ml_data/*.csv`、`models/*.pkl`、`yolo11n.pt`
