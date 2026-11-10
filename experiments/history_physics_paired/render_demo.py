"""Render the preselected held-out fatigue pair from demo_case.npz."""

import argparse
from functools import lru_cache
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont


FPS = 12
BACKGROUND = "#101925"
PANEL = "#1e2c3e"
TEXT = "#eef4fa"
MUTED = "#b7c8da"
ACCENTS = ("#88d3ff", "#ffc185")


def render(case, output, font_path):
    past, future = case["past"], case["future"]
    events, predictions = case["events"], case["predicted_events"]
    observed = Image.fromarray(case["frames"]).resize((288, 144), Image.Resampling.NEAREST)

    @lru_cache
    def font(size):
        return ImageFont.truetype(font_path, size)

    def frame(step):
        image = Image.new("RGB", (1120, 680), BACKGROUND)
        draw = ImageDraw.Draw(image)

        def text(x, y, value, size, color=TEXT, anchor="lm"):
            draw.text((x, y), value, font=font(size), fill=color, anchor=anchor)

        text(32, 28, "历史物理小实验  /  成对疲劳预测", 18, MUTED)
        text(32, 72, "现在看起来一样，接下来谁先断？", 36)
        if step == 0:
            caption = "预测起点    两块试件同一张画面，过去平均载荷相同"
        else:
            caption = f"未来第 {step:03d} 步    两块试件收到相同载荷 {future[step - 1]:.2f}"
        text(32, 116, caption, 22, MUTED)

        for side, x in enumerate((32, 570)):
            name = "AB"[side]
            accent = ACCENTS[side]
            failed = step >= events[side]
            draw.rounded_rectangle((x, 141, x + 518, 454), 15, fill=PANEL)
            text(x + 18, 167, f"试件 {name}", 24, accent)
            picture = observed
            if failed:
                picture = Image.new("RGB", observed.size, (22, 34, 50))
                picture.paste(observed.crop((10, 0, 144, 144)), (0, 0))
                picture.paste(observed.crop((144, 0, 278, 144)), (154, 0))
            image.paste(picture, (x + 115, 184))
            text(x + 259, 347, f"仿真第 {events[side]} 步断裂" if failed else "尚未断裂",
                 21, accent if failed else TEXT, "mm")
            text(x + 18, 379, "停载前 60 步载荷", 15, MUTED)
            text(x + 498, 379, f"均值 {past[side, :-1].mean():.3f}", 15, MUTED, "rm")
            draw.line((x + 18, 438, x + 498, 438), fill="#4a5b70")
            for i, load in enumerate(past[side, :-1]):
                start = x + 18 + 8 * i
                draw.rectangle((start, 437 - int(43 * load / 1.2), start + 5, 437), fill=accent)

        text(32, 481, "三种设置在起点给出的预测", 23)
        text(555, 481, "A 断裂步", 18, MUTED)
        text(728, 481, "B 断裂步", 18, MUTED)
        text(905, 481, "谁先断", 18, MUTED)
        labels = ("只看当前画面", "观测和载荷历史", "直接给真实损伤")
        for row, (label, (a, b)) in enumerate(zip(labels, predictions)):
            y = 520 + row * 44
            draw.rounded_rectangle((32, y - 17, 1088, y + 17), 7, fill=PANEL)
            text(48, y, label, 20)
            text(555, y, str(a) if a <= len(future) else "未断", 21)
            text(728, y, str(b) if b <= len(future) else "未断", 21)
            text(905, y, "并列" if a == b else ("A 先断" if a < b else "B 先断"), 20)
        text(32, 660, "合成仿真示意 · 裂纹被遮住 · 三组使用同一个推演器 · 断裂步从预测起点算起", 16, MUTED)
        return image

    last = int(events.max())
    steps = [0] * 32
    for step in range(1, last + 1):
        steps.extend([step] * (14 if step in events else 1))
    steps.extend([last] * 22)
    images = [frame(step) for step in steps]
    gif = output / "paired-fatigue-demo.gif"
    images[0].save(gif, save_all=True, append_images=images[1:],
                   duration=round(1000 / FPS), loop=0, optimize=True)
    if ffmpeg := shutil.which("ffmpeg"):
        subprocess.run([ffmpeg, "-y", "-v", "error", "-ignore_loop", "1", "-i", str(gif),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                        str(output / "paired-fatigue-demo.mp4")], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font", default="/System/Library/Fonts/STHeiti Medium.ttc",
                        help="Path to a font that supports Chinese")
    args = parser.parse_args()
    with np.load(args.output / "demo_case.npz", allow_pickle=False) as case:
        render(case, args.output, args.font)


if __name__ == "__main__":
    main()
