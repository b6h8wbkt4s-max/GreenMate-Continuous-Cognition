# GreenMate 连续状态认知工作台 (v1.0-final)

**【复现性声明】**
本项目所有测试结果均完全可复现。系统已通过锁定 `scenarios.json` 和模型版本（`rf_v1`），消除了运行时的随机性。任何评委按照以下指南，均可得到与验证记录中完全一致的 S1~S5 结构化判断结果。

本项目为面向番茄连续生长环境的轻量级状态认知与多源证据分析系统。包含 RAG 防幻觉、Agent 工具调用、传统机器学习预测及 YOLO 视觉感知。

## 环境要求
- Python 3.10+
- 可联网环境（用于调用大模型 API）

## 快速复现指南（10 分钟）

### 1. 安装依赖
在根目录打开终端，创建并激活虚拟环境：
```bash
python -m venv .venv
.\.venv\Scripts\activate   # Windows
# source .venv/bin/activate # Mac/Linux

pip install -r requirements.txt