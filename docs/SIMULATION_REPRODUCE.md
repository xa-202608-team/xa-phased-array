# 仿真与复现说明（SIMULATION_REPRODUCE）

> 一页纸说清：三级链仿真怎么跑、评审路径（judge）与完整复现（full）怎么跑、
> 产物是什么、种子与哈希怎么核。数据字典见 `docs/DATA_DICTIONARY.md`，
> 建模口径见 `docs/MODELING.md`。

## 1. 种子与可复现性

- 仿真 seed = **42**（`configs/phased_array.yaml::seed`）；每轨迹独立子流
  `seed_traj`，故障注入用独立子流（`rng_substream: 9001`）；
- 对比实验冻结 **5 seeds = 42..46**（`run_groups` 内 `seed = cfg.seed + s`）；
- 相同 seed + 相同 config 的 CPU 仿真跨 OS 逐位一致（`.gitattributes` 强制 LF）；
  GPU 训练允许 ±5% 容差（CUDA 非确定性）；
- 三级链结构冻结文档：`docs/simulation/PHYSICS_CHAIN.yaml`
  （`python scripts/generate_physics_chain.py` 再生成，
  `tests/test_simulation_reproduce_docs.py` 守护三级结构/配置键/生成函数存在性）。

## 2. 统一入口（五件套）

```bash
# 1) 源域数据准备（canonical NASA / synthetic 自动判定，报告 source_mode）
python scripts/prepare_source_data.py --output outputs/source_report.json

# 2) 三级链仿真（依次 sim_v2 subdose on + sim_v1 subdose off，写实际 config SHA256）
python scripts/generate_simulation.py --n_traj 200 --seed 42 \
    --manifest outputs/sim_manifest.json          # 加 --fast 为小规模冒烟

# 3) 单指标遥测预测（component.predict 薄包装，输出 prediction.json）
python scripts/predict_telemetry.py --telemetry T.csv --metadata M.yaml \
    --telemetry-name array_gain_db --output outputs/predict

# 3b) 通道级 GRU RUL 推理（F1 批3：消费 run_groups --export-inference-dir
#     导出的 val-best bundle；无标签读 x_ch，输出 rul_prediction.json）
python -m src.experiments.run_groups --level channel --export-inference-dir outputs/bundle ...
python -m component.predict_gru \
    --features data/features/phased_array/schema_ch_v1/target/channel_features.h5 \
    --bundle-dir outputs/bundle --output outputs/inference --stride 50

# 4) 评审用小规模端到端复现（仿真→HI→预测/基线→Schema 校验，CPU 分钟级）
bash scripts/reproduce_judge.sh  --output outputs/judge     # 或 .ps1（同参数/退出码）
# 5) 完整复现（200 轨迹 + 5 seeds；主步骤失败非零退出，已移除 || echo 吞错）
bash scripts/reproduce_full.sh  --output outputs/full       # 或 .ps1；--fast 仅调试（单 seed）
```

judge/full 产物（`--output` 目录）：

| 文件 | 内容 |
|------|------|
| `manifest.json` | 契约 manifest.schema.json：git_commit、config_sha256、数据 SHA256、seeds、环境、elapsed、status=REPRODUCE_OK |
| `metrics.json` | 契约 metrics.schema.json：本次运行实测指标（组 RMSE 均值 + 非学习基线），conclusion=NO_POSITIVE_TRANSFER_SUPPORTED（项目冻结结论，见 docs/MODELING.md §6） |
| `run.log` | 全步骤命令与输出 |
| `REPRODUCE_OK` | 仅整条流程成功时写出的哨兵 |
| `source_mode.txt` / `sim_manifest.json` / `predict/prediction.json` / `groups/` / `groups/inference_bundle/` / `inference/rul_prediction.json` / `baselines_channel.json` | 各步骤产物（full 含 P5.5 推理自检） |

**source_mode 语义**（`manifest.json` 因契约 schema 限制不设自定义字段，以
`metrics.run_id`、`source_mode.txt` 与 `run.log` 标注，三者一致）：

- `canonical_nasa` / `nasa_real`：真实 NASA MOSFET 源域（canonical H5 已就绪）；
- `synthetic`：本地无真实源域，judge/full 走合成源域 smoke——**该模式结果
  只证明管线连通，不得与正式源域结果混用或对外引用为正式指标**。

## 3. 逐步复现（等价手动拆解）

```bash
python -m src.sim.phased_array_sim --n_traj 200 --seed 42 --subdose on  --hash   # sim_v2
python -m src.sim.phased_array_sim --n_traj 200 --seed 42 --subdose off --hash   # sim_v1
python -m src.sim.build_channel_hi --report    # 通道级 HI（读 sim_v2）
python -m src.sim.build_array_hi  --report     # 服务级 HI（读 sim_v1）
python -m src.experiments.run_groups --level channel --output-dir OUT   # 8 组 × 5 seeds
python -m src.baselines.channel_baselines --out OUT/baselines_channel.json
```

一致性锚点：`--subdose on` 输出 `dynamics_id=phased_array_subdose_v2`，
`off` 输出 `dynamics_id=leo_coupled_v1`；H5 属性与 `sim_manifest.json` 的
SHA256 可交叉核对。

## 4. Docker

```bash
docker build --build-arg XA_GIT_COMMIT=$(git rev-parse HEAD) \
    [-t xa-phased-array:baseline-v0.1.0 .]        # GPU torch 默认; 无 GPU/NVIDIA 源不可达时加
#   --build-arg TORCH_EXTRA_INDEX=https://download.pytorch.org/whl/cpu 构建 CPU 变体
docker run --rm xa-phased-array:baseline-v0.1.0 verify
# 评审复现（canonical H5 与 ckpt 不烘焙进镜像，从只读挂载读取；缺失走 synthetic）：
docker run --rm -v <host-out>:/outputs \
    [-v <host-canonical-dir>:/artifacts/data:ro -v <host-ckpt-dir>:/artifacts/checkpoints:ro] \
    xa-phased-array:baseline-v0.1.0 reproduce_judge --output /outputs/judge
```

- `/artifacts/data`、`/artifacts/checkpoints`：只读挂载点（canonical H5 / 预训练
  ckpt），镜像内不烘焙任何 `.h5`/`.pt`；
- `/outputs`：唯一可写输出目录；
- `verify`：只跑代码/fixture/Schema/导入检查（pytest + schema 快照 SHA256 +
  关键模块导入），不依赖任何大数据工件。

## 5. 已知平台问题

- **Windows 退出期崩溃（0xC0000409）**：本机 conda + torch/h5py 在部分训练脚本
  （如 run_groups）main 完成、产物完整写出后，于解释器关闭阶段（atexit 之后、
  CRT/DLL 卸载）崩溃。judge/full 的 `RunLog.run` 对此采用**产物哨兵兜底**：
  退出码非零但期望产物 JSON 存在且可解析 → 记 WARNING 继续；产物缺失仍判失败。
  Linux/Docker/CI 无此现象。
- `reproduce_full --fast` 为调试模式：默认单 seed（4 轨迹小样本下 0.15 train
  比例遇个别种子会抽空，非正式语义）。

## 6. 冒烟基线证据

本仓 `docs/results_phased_array_smoke.md` 登录 smoke 口径结果（基于小规模
合成数据，全部组 RMSE≈0，仅证明管线连通）；正式 5-seed 冻结数字与其出处见
`docs/MODELING.md` §6 与 `results/public_summary.json`。
