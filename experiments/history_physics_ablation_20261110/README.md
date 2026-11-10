# 历史物理模型第二轮：状态反馈、约束与低载误报消融

这轮在第一轮结果基础上提出机制假设，重新训练全部对照，并用新的独立测试轨迹检验。源码与输出独立保存，第一轮结果保持原样。

[冻结协议](PROTOCOL.md) · [结果报告](../../research/history_physics_20261110/ablation/RESULTS.md) · [逐种子指标](../../research/history_physics_20261110/ablation/evaluation.json) · [配对差值](../../research/history_physics_20261110/ablation/paired_effects.json)

## 两层消融

第一层为完全同形、同初始化的 2×2 GRU 实验。每种模型都保留上一输出供计分/投影使用；“反馈”指是否把它输入下一步网络，不代表所有形式的显式状态表示。

| 模型名 | 上一步预测状态作为网络输入 | 物理输出投影 |
|---|---|---|
| `gru_plain` | 否，第六输入槽为零 | 无 |
| `gru_state` | 是 | 无 |
| `gru_projected` | 否，第六输入槽为零 | 有 |
| `gru_state_projected` | 是 | 有 |

第二层使用同形、同初始化的显式状态 MLP。疲劳的嵌套模型如下；`state_cubic` 是独立的机制先验干预。

| 模型名 | 状态更新 |
|---|---|
| `state_free` | 有符号增量 |
| `state_gated` | 有符号增量 × 归一化载荷；零载保持 |
| `state_positive` | 非负增量 × 归一化载荷，无上限截断 |
| `state_full` | `state_positive` 加临界值 1 截断 |
| `state_cubic` | `state_full` 的一次载荷门改为三次方 |

热任务比较四种 GRU 与 `state_free` / `state_full`；疲劳比较四种 GRU 与五种 MLP。总计 15 个任务–模型配置 × 3 种子 = 45 次训练。`state_free` 到 `state_full` 的热更新形式同时改变多个先验，应结合第一层分析，不能声称是单条约束的净作用。

## 公平性与诊断

- 沿用第一轮训练/ID 验证数据和 1,500 次更新预算，每个种子所有模型使用相同 minibatch。相同家族的权重初始化逐字节相同；家族内参数量一致。
- 新测试种子为 21003/21004/21005；先生成 256 步再截取短前缀，检验 32/64/128/256 步与停止输入后的行为。
- `diagnostics.py` 先在额外的低载验证集 21002 拟合一个变化率比例，再保存 `calibration.json`。主评测禁止在该文件生成前构造测试数据。
- 该比例校准使用额外验证信息，仅用于机制诊断，不能冒充同训练预算的主模型。
- 固定权重只取消断裂上限，首次越过阈值应保持不变；断裂后冻结未截断诊断轨迹以避免无关的后续爆炸。
- 在真实前态上测一步增量偏差，并与自由滚动时已有状态误差的反馈分开；同时比较训练和长程测试的状态–载荷覆盖。
- 所有非有限/未完成模型记录为失败，不能偷偷截断或删掉种子。物理硬约束的零违例来自实现，不代表学习发现了定律。

## 复现命令

在仓库根目录使用已有 `.venv`，无需下载模型或数据。正式实验为 CPU FP32、2 线程。使用新输出目录，以免覆盖已经冻结的结果。

```bash
.venv/bin/python -m unittest discover -s experiments/history_physics_ablation_20261110 -p 'test_*.py' -v
.venv/bin/python experiments/history_physics_ablation_20261110/train.py --output outputs/history_physics_ablation_replication
.venv/bin/python experiments/history_physics_ablation_20261110/diagnostics.py --output outputs/history_physics_ablation_replication --calibrate-only
.venv/bin/python experiments/history_physics_ablation_20261110/evaluate.py --output outputs/history_physics_ablation_replication
.venv/bin/python experiments/history_physics_ablation_20261110/diagnostics.py --output outputs/history_physics_ablation_replication
.venv/bin/python experiments/history_physics_ablation_20261110/plot_results.py --output outputs/history_physics_ablation_replication
.venv/bin/python experiments/history_physics_ablation_20261110/plot_diagnostics.py --output outputs/history_physics_ablation_replication
.venv/bin/python experiments/history_physics_ablation_20261110/audit_results.py --output outputs/history_physics_ablation_replication
```

训练核心四文件与协议在训练前固定；评测/诊断实现及其哈希在新测试评分前保存。每个运行保留 `best.pt`、`final.pt`、`trace.json` 和 `metadata.json`。正式输出位于 `outputs/history_physics_ablation_20261110/`；完整预测与失败案例可定位到模型、种子、测试条件和轨迹编号。

`write_report.py` 校验本次主评测与机制诊断的哈希后重建中文报告，避免把本次案例解释误套用到新实验。重建本次报告用 `.venv/bin/python experiments/history_physics_ablation_20261110/write_report.py`。`plot_reviewed.py` 是仅调整非负指标坐标轴范围的展示版本；冻结的原绘图脚本和原图保留。

本轮仍为已知材料参数、数值状态监督的合成实验，不包含视觉编码、视频生成或真实材料校准。
