# History-Dependent Physics Models

本地可训练的历史依赖物理模型：热积累与冷却、疲劳裂纹增长与断裂预测。

Small, reproducible experiments comparing recurrent models with explicit physical
state updates. This repository is independent of OASIS and requires no pretrained
video model, Minecraft installation, external dataset, or GPU.

## 已完成的实验

| 实验 | 设计 | 报告 |
|---|---|---|
| 本机可行性 | 小型数值与视觉网络的 CPU/MPS 训练测速 | [说明](experiments/history_physics_local_20261110/README.md) |
| 第一轮 | 热/疲劳两任务 × GRU/受约束模型 × 3 种子，12次训练 | [完整结果](research/history_physics_20261110/benchmark/RESULTS.md) |
| 第二轮 | 同形GRU状态反馈×投影；状态MLP逐级约束；45次训练 | [消融与机制诊断](research/history_physics_20261110/ablation/RESULTS.md) |

第二轮表明：显式状态反馈没有跨条件稳定的长程收益；物理约束分别改善停载保持、单调性和断裂后稳定性，但不能保证增长率正确。原完整模型在低载荷、中后期裂纹状态下高估增长率，导致部分轨迹提前断裂。

| 低载128步，3种子均值±样本标准差 | 裂纹RMSE | 断裂误报率 |
|---|---:|---:|
| 原完整状态模型 | 0.0422 ± 0.0011 | 4.57% ± 0.11% |
| 三次方载荷先验 | 0.0097 ± 0.0016 | 0%，三种子均为零 |

到256步，三次方模型仍有1.93%误报和7.29%漏报；随机载荷下也没有全面改善。
所有数字来自固定合成动力学，不是对真实材料或视觉世界模型的验证。

![实际状态条件下的增长率偏差](research/history_physics_20261110/ablation/figures/occupied_state_rate_bias.png)

## 环境与复现

记录环境为 Python 3.12.13、Apple M2 Pro、16 GiB内存。正式数值实验用CPU FP32、2线程，第一轮训练/验证合计约120秒，第二轮约398秒。不同硬件与库实现可能产生数值差异。

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r experiments/history_physics_requirements.txt

# 第一轮：独立训练、评测、绘图和审计。
.venv/bin/python -m unittest discover -s experiments/history_physics_benchmark_20261110 -p 'test_*.py' -v
.venv/bin/python experiments/history_physics_benchmark_20261110/train.py --output outputs/history_physics_benchmark_20261110
.venv/bin/python experiments/history_physics_benchmark_20261110/evaluate.py --output outputs/history_physics_benchmark_20261110
.venv/bin/python experiments/history_physics_benchmark_20261110/plot_results.py --output outputs/history_physics_benchmark_20261110
.venv/bin/python experiments/history_physics_benchmark_20261110/audit_results.py --output outputs/history_physics_benchmark_20261110

# 第二轮：先锁定独立验证集校准，再触碰新测试集。
.venv/bin/python -m unittest discover -s experiments/history_physics_ablation_20261110 -p 'test_*.py' -v
.venv/bin/python experiments/history_physics_ablation_20261110/train.py --output outputs/history_physics_ablation_20261110
.venv/bin/python experiments/history_physics_ablation_20261110/diagnostics.py --output outputs/history_physics_ablation_20261110 --calibrate-only
.venv/bin/python experiments/history_physics_ablation_20261110/evaluate.py --output outputs/history_physics_ablation_20261110
.venv/bin/python experiments/history_physics_ablation_20261110/diagnostics.py --output outputs/history_physics_ablation_20261110
.venv/bin/python experiments/history_physics_ablation_20261110/plot_reviewed.py --output outputs/history_physics_ablation_20261110
.venv/bin/python experiments/history_physics_ablation_20261110/plot_diagnostics.py --output outputs/history_physics_ablation_20261110
.venv/bin/python experiments/history_physics_ablation_20261110/audit_results.py --output outputs/history_physics_ablation_20261110
```

训练器拒绝覆盖已有冻结运行。上面的命令面向新clone，原始记录的公开副本保存在`research/`，不会与新生成的`outputs/`混用。两轮训练数据相同、测试种子不同；第二轮审计会核对第一轮数据manifest，因此按上面顺序运行。

详细设计见第一轮[协议](experiments/history_physics_benchmark_20261110/PROTOCOL.md)和第二轮[协议](experiments/history_physics_ablation_20261110/PROTOCOL.md)。正式训练源码保留原始字节与哈希；两轮分别通过14/19项自动检查，独立审计也已记录。报告生成器的案例文字绑定原始运行，不会把原结果套用到新训练。

## 原始结果附件与审计边界

仓库包含代码、报告、指标JSON、图表与审计记录。完整逐步预测、诊断数组和模型权重在[Release附件](https://github.com/JANZeee/history-dependent-physics/releases/tag/history-physics-2026-11-10)，包含832个结果文件。

下载`history-physics-20261110-evidence.tar.gz`及同名`.sha256`文件，在单独目录验证并解压：

```bash
shasum -a 256 -c history-physics-20261110-evidence.tar.gz.sha256
mkdir evidence-snapshot
tar -xzf history-physics-20261110-evidence.tar.gz -C evidence-snapshot
```

公开副本已将本机绝对路径改为相对路径；NPZ数值数组保持逐字节相同，checkpoint张量值保持相同。修改路径会改变JSON及checkpoint文件哈希，记录中的既有审计PASS与冻结哈希对应发布前的本地原件。请使用[发布manifest](research/history_physics_20261110/PUBLICATION_MANIFEST.json)核对原文件、公开附件与仓库副本的SHA256映射；不要把改写后的文件当成具有原哈希的对象。完整端到端复核应在新的输出目录重新训练与评测。

本仓库保留失败和负面结果，不声称模型已具备真实材料可靠性、通用物理推理或视频生成物理一致性。
