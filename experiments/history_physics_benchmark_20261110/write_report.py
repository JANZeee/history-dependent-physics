"""Render a Chinese research note from completed benchmark artifacts; no training."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("outputs/history_physics_benchmark_20261110"))
    args = p.parse_args()
    raw = (args.output / "evaluation.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != "4a6c210c779b2f8ba0775bcd7234c7a0d9847ddcaa28ca73c288a457560d683c":
        raise RuntimeError("Case narrative is specific to the recorded experiment; use evaluation.json and plot_results.py for a new run")
    report = json.loads(raw)
    manifest = json.loads((args.output / "training_manifest.json").read_text())
    training = json.loads((args.output / "training_complete.json").read_text())
    lookup = {(r["task"], r["model"], r["condition"]): r["metrics"] for r in report["aggregate"]}

    def rec(task, model, condition, metric):
        return lookup[(task, model, condition)][metric]

    def cell(task, model, condition, metric, precision=4, scale=1):
        v = rec(task, model, condition, metric)
        return f"{v['mean']*scale:.{precision}f} ± {v['sample_sd']*scale:.{precision}f}"

    def table(specs):
        rows = ["| 指标（越小越好） | 普通 GRU | 受约束状态模型 |", "|---|---:|---:|"]
        for label, task, condition, metric, precision, scale in specs:
            rows.append("| " + label + " | " + " | ".join(cell(task, m, condition, metric, precision, scale) for m in ("gru", "constrained")) + " |")
        return "\n".join(rows)

    main_specs = [
        ("热：32 步 RMSE / K", "thermal", "random_32", "state_rmse", 3, 1),
        ("热：64 步 RMSE / K", "thermal", "random_64", "state_rmse", 3, 1),
        ("热：128 步 RMSE / K", "thermal", "random_128", "state_rmse", 3, 1),
        ("疲劳：32 步 RMSE", "fatigue", "random_32", "state_rmse", 4, 1),
        ("疲劳：64 步 RMSE", "fatigue", "random_64", "state_rmse", 4, 1),
        ("疲劳：128 步 RMSE", "fatigue", "random_128", "state_rmse", 4, 1),
        ("疲劳：128 步、真实断裂前 RMSE", "fatigue", "random_128", "pre_failure_rmse", 4, 1),
        ("断裂：128 步、全样本截断事件步数 MAE", "fatigue", "random_128", "event_time_capped_mae_all_steps", 3, 1),
        ("断裂：128 步漏报率 / %", "fatigue", "random_128", "false_negative_rate", 2, 100),
    ]
    passive_specs = [
        ("冷却尾段温度 RMSE / K", "thermal", "passive_tail_128", "tail_state_rmse", 3, 1),
        ("冷却结束温度 MAE / K", "thermal", "passive_tail_128", "tail_final_mae", 3, 1),
        ("尾段总降温量 MAE / K", "thermal", "passive_tail_128", "tail_cooling_change_mae_K", 3, 1),
        ("冷却尾段出现局部升温的步数比例 / %", "thermal", "passive_tail_128", "tail_passive_increase_fraction", 2, 100),
        ("冷却尾段低于环境温度的步数比例 / %", "thermal", "passive_tail_128", "tail_ambient_undershoot_fraction", 2, 100),
        ("无载荷尾段裂纹 RMSE", "fatigue", "passive_tail_128", "tail_state_rmse", 4, 1),
        ("无载荷尾段最终绝对漂移（相对自身第 32 步）", "fatigue", "passive_tail_128", "tail_own_state_final_change_abs_mean", 4, 1),
        ("随机 128 步中裂纹缩短的步数比例 / %", "fatigue", "random_128", "damage_decrease_fraction", 2, 100),
    ]
    low_specs = [
        ("低载荷 128 步裂纹 RMSE", "fatigue", "low_load_128", "state_rmse", 4, 1),
        ("低载荷 128 步断裂误报率 / %", "fatigue", "low_load_128", "false_positive_rate", 2, 100),
        ("低载荷全样本截断事件步数 MAE", "fatigue", "low_load_128", "event_time_capped_mae_all_steps", 3, 1),
    ]
    total_seconds = sum(r["training_seconds"] for r in training["runs"])
    rss_mib = max(r["process_max_rss_bytes"] for r in training["runs"]) / 1024**2
    parts = [f"""# 本地历史物理实验结果（2026-11-10）

已完成两项任务、两种模型、三个训练种子的全部 12 次训练。受约束状态模型在 4 倍训练长度的轨迹上误差更小，停止加热后持续冷却、停止载荷后保存损伤；但在低载荷长序列中，它的断裂误报率明显高于 GRU。**保持物理约束不等于准确预测断裂。**

这是可复现的合成数值实验：不包含图像编码器、视频生成器或真实材料数据。裂纹状态是临界裂纹长度归一化后的 a，a=1 代表断裂；其误差没有温度的 K 单位。以下 ± 均为 3 个训练种子的样本标准差，不是置信区间；三次训练共享数据集。

## 实验设置与公平性

- 训练种子 101、202、303。每项任务 5,000 条训练轨迹、512 条独立验证轨迹，长度均为 32。
- 每次训练 1,500 次 Adam 更新，batch=64，lr=0.001，梯度裁剪=1。相同种子的两个模型使用完全相同 minibatch 索引。均以每一步状态的 MSE 为目标，不使用 teacher forcing。
- 每 100 次更新在验证集评分，只按验证 MSE 选择 checkpoint。所有 12 次训练结束后才构造正式测试集；未根据测试结果重训或调参。
- 输入信息相同：当前控制量、时间/循环增量、材料参数、初始条件。GRU 用 64 维隐状态；受约束模型用显式预测状态和两层 112 宽 MLP 学习变化率。两者都不能读取当前真实状态。
- GRU 13,697 参数；热受约束模型 13,666，疲劳受约束模型 13,553。参数量差异小于 1.1%。本轮匹配更新次数，不匹配运行时间/FLOPs。
- 结构先验与状态表示同时改变，不能将差异解释为单独某一条约束的因果效果。两种模型未分别进行超参数搜索；这是固定预算基线比较。

本机为 Apple M2 Pro / 16 GiB 统一内存。实际运行使用 CPU FP32、2 线程；Python {manifest['python']}，PyTorch {manifest['torch']}。12 次训练和定期验证的计时之和为 **{total_seconds:.2f} 秒**，正式评测约 **{report['evaluation_seconds']:.2f} 秒**；训练进程累计峰值 RSS 为 **{rss_mib:.1f} MiB**，不是整机内存占用。初始化、部分保存和绘图未计入训练计时。

## 参考物理与模型先验

热参考满足 C·d(T−20)/dt=P−H·(T−20)，每个控制步用解析指数解生成。C∈[80,120] J/K，H∈[0.5,4] W/K，P∈[0,200] W（35% 停热），dt∈[0.5,5] 秒，初温 20°C。

热模型学习非负热源和散热率，更新使用指数衰减形式；它没有直接调用参考 H/C。该形式仍是强物理先验：保证当前温度不低于环境时，无输入不会反弹升温，但不能保证学到正确的散热时间常数。

疲劳参考为无量纲 Paris 型稳定增长 da/dN=k·s³·a^(3/2)，用解析步进计算并在 a=1 吸收。a₀∈[0.08,0.2]、k∈[0.04,0.12]、s∈[0.3,1.5]（25% 停载）、dN∈[0.5,1.5]。假设固定最大应力而变化应力范围，以使用固定临界裂纹阈值。未建模裂纹闭合、塑性和失稳前加速；不是金属反复大幅折叠或橡皮筋拉断的真实校准模型。

疲劳模型学习非负有限步增量，载荷为零时增量严格为零，a 达到 1 后保持 1；没有硬编码 s³ 或 a^(3/2)。这些单调、停载保持和断裂吸收性质来自架构。

## 长时程泛化

每项任务固定 512 条独立测试轨迹。32、64、128 步使用同一批 128 步序列的嵌套前缀。训练只见过长度 32；128 步是 4 倍时程外推，材料参数仍在训练范围。

{table(main_specs)}

随机疲劳测试在 32/64/128 步分别有 41/302/497 条真实断裂轨迹（总数 512）。128 步大量状态已吸收在 1，整体 RMSE 容易掩盖断裂前误差，因此同时报告真实断裂前 RMSE，并增加下方几乎不发生断裂的低载荷测试。

事件指标使用第一个预测 a≥0.9999 的观测步；参考事件使用 a=1。未发生事件统一记为 horizon+1，故这里的 MAE 是**包含未发生/漏报事件的截断观测步误差**，不是连续秒数或疲劳循环寿命误差。128 步真实断裂中的漏报为 GRU 217/497、238/497、230/497；受约束模型每个种子均为 1/497。

随机 128 步仅有 15 条真实未断裂轨迹：GRU 误报为 0/15；受约束模型为 4/15、6/15、1/15。这个误报率分母很小，不能据此稳定估计稀有事件性能。

![长时程与被动尾段结果](figures/aggregate_metrics.png)

## 冷却恢复与损伤不可逆性

独立 512 条序列：前 32 步允许输入，此后连续 96 步输入为零。疲劳前缀使用训练支持内的低幅载荷 [0.3,0.8]，防止先全部断裂再平凡地保持 1；所有 512 条在开始停载时都未断裂。

{table(passive_specs)}

物理违例容差为归一化状态 1e−5（热为 0.001 K）。比例按状态转移步统计，而非“出现任何错误的轨迹比例”。冷却尾段每个种子分母 512×96=49,152；随机 128 步裂纹缩短分母 65,536。

GRU 整体仍在冷却，但会出现局部非物理反弹，不能表述成“停止加热后总体升温”。疲劳无载荷尾段中，GRU 相对自身第 32 步的预测平均降低约 0.04249，构成模型内的损伤恢复；即使这种漂移偶然靠近真值，它仍违反不可逆性。受约束模型严格不漂移，但会一直保留第 32 步的估计偏差。

受约束模型的零违例是结构性保证，不应作为“从数据发现了物理定律”的证据。本轮的冷却恢复仅指温度回落，未测试重新加热、制动性能恢复或材料修复。

## 主要负面结果：低载荷断裂误报

低载荷测试使用独立的 512 条 128 步序列，s∈[0.3,0.8]，每一步幅值仍处于训练支持范围。只有 1 条真实断裂，另外 511 条应判为未断裂。

{table(low_specs)}

| 种子 | GRU 误报 / 真实未断裂 | 受约束模型误报 / 真实未断裂 |
|---|---:|---:|
| 101 | 3 / 511 | 32 / 511 |
| 202 | 5 / 511 | 33 / 511 |
| 303 | 2 / 511 | 32 / 511 |

受约束模型在低载荷下依然有更低的状态 RMSE，却更常越过断裂阈值；不能据平均状态误差宣称事件预测全面优于 GRU。非负增量累积可能持续放大正偏差，这是待验证的解释，本次未做因果消融。

唯一真实断裂样本的受约束事件误差为 38、37、40 步；样本太少，不能据此估计总体漏报能力。这里的全样本事件 MAE 被 511 条未断裂样本主导，也必须与误报计数一起读。

![低载荷误报与对照失败轨迹（测试后描述性选例）](figures/low_load_failure.png)

## 可复查的失败案例

正式协议事先规定：每个模型、每个种子分别选最大 128 步 RMSE、最大无输入尾段 RMSE，以及疲劳最大事件步误差。共保存 30 个带选择标签的案例文件；每个案例同时包含两个模型预测，可能重复选中同一轨迹。以下是这些规则下的例子，样本编号从 0 开始：

| 条件、种子、样本 | 参考行为 | 观察到的失败 |
|---|---|---|
| 热停输入，202，153 | 第 32 步 98.17°C，最终 26.18°C | GRU 最终 40.01°C，尾段 RMSE 14.64 K；受约束最终 24.34°C |
| 热停输入，101，486 | 第 32 步 81.82°C，最终 31.05°C | 受约束冷却过快，最终 25.16°C、尾段 RMSE 7.01 K；同例 GRU 尾段 RMSE 2.83 K |
| 疲劳停输入，303，416 | 裂纹保持 0.2919 | GRU 从自己第 32 步 0.6781 回落至 0.2573；受约束保持 0.3088，仍有估计偏差 |
| 疲劳随机，202，394 | 第 30 步断裂 | GRU 到第 128 步都未报断裂，最终 0.9548；受约束第 31 步报告 |
| 疲劳随机，101，6 | 第 84 步断裂 | 受约束第 61 步报告，提前 23 步；GRU 在 128 步内未报告 |

低载荷补充图的案例是在看完测试结果后按记录在 `figures/low_load_failure_selection.json` 中的规则选出，只用于解释误报，不属于事先规定的 30 个案例，也没有参与训练或调参。图中 seed 303、样本 445 的参考裂纹在第 128 步为 0.9771，尚未断裂；受约束模型在第 87 步即报断裂并保持 1。同例 GRU 第 61 步也曾越过阈值，之后又回落，因此不能把这个样本描述为“GRU 预测正确”。

全部图： [长程最差案例](figures/worst_long_rmse.png)、[被动尾段最差案例](figures/worst_passive_tail_rmse.png)、[断裂事件最差案例](figures/worst_event_time.png)。对应 PDF 也已保存。

## 结论与后续边界

本轮支持的结论是：在这两种已知参数的合成动力学中，小型显式状态模型能在有限本地算力上学到较准确的长程数值预测，同时满足预设的冷却与不可逆性约束。低载荷误报和散热过快证明约束不充分，仍需验证变化率与临界事件的准确性。

下一轮优先把“状态表示”和“约束”分开消融：保留同一显式状态网络，比较无约束增量、仅非负增量、完整停载/吸收约束；另设 GRU 加相同输出约束的基线。低载荷校准、事件目标或更长训练历史应使用新训练/验证集和新冻结测试集。本次测试集不能用于调整阈值后再充当未见测试集。

尚未证明：真实材料可迁移性、未知材料参数推断、仅从画面识别隐状态、视频生成的物理一致性，以及约束相对其他强基线的普遍优势。三种种子也不足以支持宽泛的统计显著性主张。

## 复现、审计与原始文件

14 项自动检查覆盖参考解析解、半步一致性、数据前缀、模型因果性/有限梯度/硬约束、事件删失计分及被动尾段定义。另从 54 份保存的预测 NPZ 独立重算指标并审计 minibatch 与验证选模：1,813 项核验、0 项不一致；审计详情见 `audit_results.json`。

训练指纹：`{manifest['fingerprint']}`。`training_manifest.json` 锁定配置与核心源文件哈希，数据 manifest 保存训练/验证/测试数组哈希，`evaluation.json` 保存评测源码哈希与训练完成文件哈希。

全部 checkpoint、逐步预测与训练曲线均在本结果目录；源码及复现命令见 [实验 README](../../experiments/history_physics_benchmark_20261110/README.md)。正式结果的完整逐种子/均值/样本标准差/分母见 [evaluation.json](evaluation.json)。

## 关键指标逐种子明细

| 指标 | 模型 | seed 101 | seed 202 | seed 303 |
|---|---|---:|---:|---:|
"""]
    for label, task, condition, metric, precision, scale in main_specs + passive_specs + low_specs:
        for model in ("gru", "constrained"):
            values = rec(task, model, condition, metric)["per_seed"]
            nums = " | ".join(f"{values[str(s)]*scale:.{precision}f}" for s in (101, 202, 303))
            parts.append(f"| {label} | {'GRU' if model == 'gru' else '受约束'} | {nums} |\n")
    parts.append("\n## 训练运行明细\n\n| 任务 | 模型 | 种子 | 参数量 | 验证选择更新步 | 训练及验证秒数 |\n|---|---|---:|---:|---:|---:|\n")
    for r in training["runs"]:
        parts.append(f"| {r['task']} | {r['model']} | {r['seed']} | {r['parameters']} | {r['best_step']} | {r['training_seconds']:.2f} |\n")
    path = args.output / "RESULTS.md"
    path.write_text("".join(parts))
    print(path.resolve())


if __name__ == "__main__":
    main()
