# 本地历史物理实验：热积累、冷却、疲劳与断裂

已在本机完成 2 个任务 × 2 种模型 × 3 个训练种子，共 12 次训练。
这是数值状态预测的最小实验，输入为动作、时间/循环增量、已知材料参数和初始条件；本轮不训练视频生成器。

[完整结果与失败案例](../../research/history_physics_20261110/benchmark/RESULTS.md) ·
[冻结实验协议](PROTOCOL.md) ·
[逐种子指标](../../research/history_physics_20261110/benchmark/evaluation.json)

## 文件与职责

| 文件 | 用途 |
|---|---|
| `data.py` | 热方程与归一化 Paris 裂纹增长方程的解析步进、独立数据集 |
| `models.py` | 普通 GRU；显式非负温差/不可逆裂纹状态模型 |
| `train.py` | 固定预算、配对 minibatch、仅验证集选 checkpoint、源码哈希 |
| `evaluate.py` | 32/64/128 步、96 步停止输入、低载荷测试及预定最差案例 |
| `plot_results.py` | 正式指标和所有预定失败案例图 |
| `plot_supplementary.py` | 测试后描述性选例：低载荷误报与 GRU 漏报 |
| `write_report.py` | 从本次冻结结果生成中文报告（含本次案例解读，校验评测哈希） |
| `audit_results.py` | 从逐条预测独立重算指标并审计配对训练 |
| `test_*.py` | 解析参考、约束、因果性、删失事件计分的 14 项检查 |

## 复现

在仓库根目录执行；现有 `.venv` 已包含 Python 3.12.13、PyTorch 2.12.1、NumPy 和 Matplotlib。
正式运行使用 CPU FP32、2 个线程，未使用 MPS。无需下载预训练模型或数据。

```bash
.venv/bin/python -m unittest discover -s experiments/history_physics_benchmark_20261110 -p 'test_*.py' -v

# 使用全新输出目录；训练器拒绝覆盖已经冻结的实验。
.venv/bin/python experiments/history_physics_benchmark_20261110/train.py --output outputs/history_physics_benchmark_replication
.venv/bin/python experiments/history_physics_benchmark_20261110/evaluate.py --output outputs/history_physics_benchmark_replication
.venv/bin/python experiments/history_physics_benchmark_20261110/plot_results.py --output outputs/history_physics_benchmark_replication
.venv/bin/python experiments/history_physics_benchmark_20261110/plot_supplementary.py --output outputs/history_physics_benchmark_replication
.venv/bin/python experiments/history_physics_benchmark_20261110/audit_results.py --output outputs/history_physics_benchmark_replication
```

`write_report.py` 的案例文字专属于本次实测，校验原始 `evaluation.json` 哈希后才生成报告；它不会将旧案例文字套用到新的复现实验。需要重建本次报告时执行 `.venv/bin/python experiments/history_physics_benchmark_20261110/write_report.py`。复现实验的指标和图由上述通用评测/绘图脚本生成。

正式结果位于 `outputs/history_physics_benchmark_20261110/`。每次运行保存 `best.pt`、`final.pt`、`trace.json` 和 `metadata.json`；输出根目录保存训练/数据哈希、全部预测、30 个预定最差案例及图表。指标中的 `null` 表示分母为零，不表示误差为零。

训练配置与 `data.py`、`models.py`、`train.py`、`PROTOCOL.md` 已冻结。修改这些文件需要新建实验协议和输出目录；不能复用本次结果声称是修改后的模型。报告和画图脚本仅消费已完成的评测结果，不触发训练。

本轮按参数量、输入信息、训练样本、优化步骤与验证规则配对，但没有匹配 FLOPs，也没有对两个模型分别做超参数搜索。显式状态架构与物理先验同时改变，因此不能把差异全部归因于某一条约束。
