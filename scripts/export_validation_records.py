"""B08 验证记录导出脚本：把 S1~S5 正式验证的 run_id 从 analysis_runs 表导出为 docs/B08_Validation_Records.md。

用法：python scripts/export_validation_records.py
"""
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import db  # noqa: E402

OUT_PATH = ROOT / "docs" / "B08_Validation_Records.md"

# S1~S5 正式验证 run_id（按场景顺序，以库内实际记录为准）
RUN_IDS = {
    "S1": "bdf4fae1-d813-484a-ac70-dfc0ff2c441c",
    "S2": "aba5a46a-5773-4c34-819b-4f7228e0ee8c",
    "S3": "72e880fd-09de-4539-a2ac-6eebce5402e8",
    "S4": "034831d9-98d8-47b2-8018-4f2e04d2c73a",
    "S5": "f7a78da4-fcf4-4ad8-9a71-c6b7781e53fd",
}

STATUS_MARK = {"ok": "✅ ok", "unavailable": "⚠️ unavailable", "error": "❌ error"}


def _summary(e):
    """每条证据的摘要列。"""
    r = e.get("result")
    if e["status"] != "ok":
        return e.get("error") or "—"
    t = e["source_type"]
    if t == "B01_STATE":
        return f"{r['name']}：{r['temperature']}°C / {r['humidity']}% / {r['light']} lux"
    if t == "B02_HISTORY":
        note = f"；{r['scenario_note']}" if r.get("scenario_note") else ""
        return f"{r['count']} 点（{r['start']} ~ {r['end']}）{note}"
    if t == "B03_WEATHER":
        tmax = round(max(r["temperature_2m"]), 1)
        pmax = round(max(r["precipitation_probability"]), 1)
        return f"24h 逐小时；最高 {tmax}°C；降雨概率最高 {pmax}%"
    if t == "B04_KNOWLEDGE":
        note = f"；{r['scenario_note']}" if r.get("scenario_note") else ""
        return f"{r['count']} 条检索结果{note}"
    if t == "B06_ML_PREDICT":
        base = (
            f"预测 {r['predicted_attention']}（模型 {r.get('model_version')}，"
            f"窗口截至 {r.get('window_end')}）"
        )
        if r.get("basis"):
            base += f"；{r['basis']}"
        return base
    if t == "B06_VISION":
        return r.get("description") or json.dumps(r, ensure_ascii=False)[:80]
    return "—"


def _limitations(run, evidence):
    """最终的限制条件说明：不完整证据 + LLM 受限结论 + 证据分歧。"""
    j = json.loads(run["judgment"]) if isinstance(run["judgment"], str) else run["judgment"]
    comp = j.get("evidence_completeness") or {}
    ex = run.get("explanation") or ""
    lines = []
    if comp.get("total", 0) > comp.get("ok", 0):
        bad = [e["source_type"] for e in evidence if e["status"] != "ok"]
        lines.append(f"证据不完整（{comp.get('ok')}/{comp.get('total')}）：{'、'.join(bad)} 未正常返回")
        for sent in ex.replace("。", "。\n").split("\n"):
            if "受限" in sent:
                lines.append("LLM 结论：" + sent.strip())
    for c in j.get("evidence_conflicts") or []:
        lines.append("证据分歧：" + c)
    if not lines:
        lines.append("无限制条件：证据完整（6/6），无冲突")
    return lines


def run_section(scenario, run):
    j = json.loads(run["judgment"]) if isinstance(run["judgment"], str) else run["judgment"]
    evidence = json.loads(run["evidence"]) if isinstance(run["evidence"], str) else run["evidence"]
    env = j.get("current_environment") or {}
    trend = j.get("recent_trend") or {}
    pred = j.get("next_window_prediction") or {}
    comp = j.get("evidence_completeness") or {}

    env_ok = next((e for e in evidence if e["source_type"] == "B01_STATE" and e["status"] == "ok"), None)
    input_row = ""
    if env_ok:
        s = env_ok["result"]
        input_row = f"| {s['name']} | {s['temperature']} °C | {s['humidity']} % | {s['light']} lux |"

    lines = [
        f"## {scenario} {j.get('scenario_name') or ''}",
        "",
        f"- **run_id**：{run['run_id']}",
        f"- **分析时刻**：{run['analysis_time']} ｜ **入库时间**：{run['created_at']}",
        f"- **地点**：{run['location']}",
        "",
        "### 输入状态（B01_STATE）",
        "",
        "| 状态 | 温度 | 湿度 | 光照 |",
        "|---|---|---|---|",
        input_row,
        "",
        "### 工具执行状态（证据链）",
        "",
        "| 证据 | 来源 | 状态 | 摘要 |",
        "|---|---|---|---|",
    ]
    for e in evidence:
        lines.append(
            f"| {e['source_type']} | {e['source']} | {STATUS_MARK.get(e['status'], e['status'])} | {_summary(e)} |"
        )
    lines += [
        "",
        f"**完整度：{comp.get('ok', 0)}/{comp.get('total', 0)}"
        f"（ok {comp.get('ok', 0)} · unavailable {comp.get('unavailable', 0)}"
        f" · error {comp.get('error', 0)}）**",
        "",
        "### 结构化判断",
        "",
        f"- **当前环境判断**：{env.get('assessment') or '（无）'}",
        f"- **近期趋势**：{trend.get('assessment') or '（无）'}",
    ]
    pred_line = f"- **下一窗口预测**：{pred.get('predicted_attention') or 'UNKNOWN'}"
    if pred.get("model_version"):
        pred_line += f"（模型 {pred['model_version']}"
        if pred.get("window_end"):
            pred_line += f"，窗口截至 {pred['window_end']}"
        pred_line += "）"
    lines.append(pred_line)
    if pred.get("basis"):
        lines.append(f"  - 提升依据：{pred['basis']}")
    if j.get("evidence_conflicts"):
        lines.append("- **证据冲突/分歧**：")
        for c in j["evidence_conflicts"]:
            lines.append(f"  - ⚠ {c}")
    else:
        lines.append("- **证据冲突/分歧**：（无）")
    lines += [
        "",
        "### 限制条件说明",
        "",
    ]
    for lim in _limitations(run, evidence):
        lines.append(f"- {lim}")
    if run.get("explanation"):
        lines += [
            "",
            "### LLM 解释（原文）",
            "",
            "> " + run["explanation"].replace("\n", "\n> "),
        ]
    return "\n".join(lines) + "\n"


def main():
    cfg = json.loads((ROOT / "scenarios.json").read_text(encoding="utf-8"))
    header = [
        "# B08 正式验证记录（S1~S5 物理留存）",
        "",
        f"> 导出时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ｜ "
        f"数据来源：sensor.db `analysis_runs` 表 ｜ "
        f"导出脚本：scripts/export_validation_records.py",
        f"> 对应版本：app_version {cfg['app_version']} ｜ model_version {cfg['model_version']}",
        "",
        "---",
        "",
    ]
    sections = []
    for scenario, rid in RUN_IDS.items():
        run = db.get_analysis_run(rid)
        if not run:
            sections.append(f"## {scenario}（缺失）\n\n> 未在 analysis_runs 中找到 run_id：{rid}\n")
            continue
        sections.append(run_section(scenario, run))
    OUT_PATH.write_text("\n".join(header) + "\n".join(sections), encoding="utf-8")
    print(f"written: {OUT_PATH} ({len(RUN_IDS)} scenarios)")


if __name__ == "__main__":
    main()
