"""生成连续多天的模拟环境数据并写入 SQLite。

数据特征：
- 正常昼夜变化：温度 14 时最高、凌晨最低；湿度与温度反相；光照白天正弦、夜间为 0
- 随机插入 4-6 个连续扰动事件（温度突升、湿度下降，持续数小时后恢复）
- 所有数据 source_type = SIMULATED

重复执行会先清空旧的 SIMULATED 数据再写入（幂等）。

用法：python simulate_data.py [--days 7] [--interval-minutes 30] [--seed 42]
"""
import argparse
import math
import random
from datetime import datetime, timedelta

import db

SOURCE_TYPE = "SIMULATED"


def generate(days, interval_minutes, seed):
    rng = random.Random(seed)
    step = timedelta(minutes=interval_minutes)
    end = datetime.now().replace(second=0, microsecond=0)
    # 对齐到整点间隔，保证时间戳规整（如 08:00 / 08:30）
    end -= timedelta(minutes=end.minute % interval_minutes)
    start = end - timedelta(days=days) + step
    total = int(days * 24 * 60 / interval_minutes)

    # 基线昼夜曲线
    temps, hums, lights = [], [], []
    for i in range(total):
        t = start + i * step
        hour = t.hour + t.minute / 60
        # 温度：14 时最高 ~29°C，凌晨 2 时最低 ~15°C，加小幅噪声
        temp = 22 + 7 * math.cos(2 * math.pi * (hour - 14) / 24) + rng.uniform(-0.3, 0.3)
        # 湿度：与温度反相，白天 ~42%，夜间 ~68%
        hum = 55 - 13 * math.cos(2 * math.pi * (hour - 14) / 24) + rng.uniform(-1.5, 1.5)
        # 光照：6-18 时正弦，正午峰值 ~25000 lux，夜间为 0
        light = 0.0
        if 6 <= hour <= 18:
            light = 25000 * math.sin(math.pi * (hour - 6) / 12)
        light = max(0.0, light + rng.uniform(-150, 150))
        temps.append(temp)
        hums.append(hum)
        lights.append(light)

    # 随机扰动事件：4-6 个，互不重叠；梯形窗口（前后约 1 小时过渡）
    occupied = []
    events = []
    event_count = rng.randint(4, 6)
    attempts = 0
    while len(events) < event_count and attempts < 200:
        attempts += 1
        duration_points = max(3, int(rng.uniform(3, 8) * 60 / interval_minutes))
        start_i = rng.randrange(total - duration_points)
        end_i = start_i + duration_points
        if any(start_i < o_end and end_i > o_start for (o_start, o_end) in occupied):
            continue  # 与已有事件重叠，重新采样
        occupied.append((start_i, end_i))
        d_temp = rng.uniform(4, 10)
        d_hum = rng.uniform(-12, -5)
        ramp = 2  # 前后各 2 个点（约 1 小时）线性过渡
        for j in range(start_i, end_i):
            if j - start_i < ramp:
                w = (j - start_i + 1) / ramp
            elif end_i - 1 - j < ramp:
                w = (end_i - j) / ramp
            else:
                w = 1.0
            temps[j] += d_temp * w
            hums[j] += d_hum * w
        events.append((start_i, end_i, d_temp, d_hum))

    rows = []
    for i in range(total):
        t = start + i * step
        rows.append((
            t.strftime("%Y-%m-%d %H:%M:%S"),
            round(temps[i], 2),
            round(max(10.0, min(95.0, hums[i])), 2),
            round(lights[i], 1),
            SOURCE_TYPE,
        ))
    return rows, events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=7, help="生成天数，默认 7")
    parser.add_argument("--interval-minutes", type=int, default=30, help="采样间隔（分钟），默认 30")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认 42（可复现）")
    args = parser.parse_args()

    db.init_db()
    rows, events = generate(args.days, args.interval_minutes, args.seed)
    db.replace_simulated(rows)
    print(f"已写入 {len(rows)} 条 SIMULATED 数据到 {db.DB_PATH}")
    print(f"时间范围: {rows[0][0]} ~ {rows[-1][0]}")
    print(f"扰动事件: {len(events)} 个")
    for i, (s, e, d_temp, d_hum) in enumerate(events, 1):
        print(f"  #{i} {rows[s][0]} ~ {rows[e - 1][0]}  温度 +{d_temp:.1f}°C  湿度 {d_hum:.1f}%")


if __name__ == "__main__":
    main()
