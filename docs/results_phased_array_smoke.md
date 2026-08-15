# 对比实验结果 — 相控阵天线 (plan Phase 6 / §8)

**注意: 基于 smoke/合成数据, 非最终数字; 正式迁移增益待真实数据预训练**
每组 1 个随机种子, 报 mean±std。RMSE/PHM/MAE **仅失效轨迹** (P0-2: 删失无精确 RUL); 删失轨迹报下界违反率。

| 实验组 | RMSE (mean±std) | PHM Score | MAE | 删失违反率 |
|--------|-----------------|-----------|-----|-----------|
| CH Target-only (TCN) *(M7)* | 0.0000±0.0000 | 0.00 | 0.0000 | 0.000 |
| CH Target-only (GRU) *(M7)* | 0.0000±0.0000 | 0.00 | 0.0000 | 0.000 |
| CH Source+Frozen *(M7)* | 0.0000±0.0000 | 0.00 | 0.0000 | 1.000 |
| **CH Source+MMD+Physics** *(M7)* | 0.0000±0.0000 | 0.00 | 0.0000 | 1.000 |
| CH Random+Frozen *(M7)* | 0.0000±0.0000 | 0.00 | 0.0000 | 0.000 |
| CH Random+Full+MMD *(M7)* | 0.0000±0.0000 | 0.00 | 0.0000 | 0.000 |
| CH Random+Full+NoMMD *(M7)* | 0.0000±0.0000 | 0.00 | 0.0000 | 0.000 |
| Cross-level (旧服务级) *(层级消融)* | 0.0000±0.0000 | 0.00 | 0.0000 | 0.000 |

**验收**:
- 迁移增益 (ch_target_only_tcn − 主迁移 ch_source_mmd_physics RMSE, 组均值差): +0.0000
- 配对 ΔRMSE (target − 主迁移 ch_source_mmd_physics, per-seed) [探索性分析, n<10 不作确认性结论]: +0.0000 ± 0.0000, 正向 seed 0/1, 95% CI [t(1)] [+0.0000, +0.0000] → 不显著 (CI 跨 0)
  - 注: CI 用样本 std + t(n-1) 临界值 (小样本校正); n<10 时仅供方向性参考, 不作严格假设检验结论。
  - 各迁移组配对统计 (探索性分析):
- 实现说明 (组名 ↔ 训练协议):
  · source_pretrain_finetune: 加载源 ckpt + **encoder 冻结** (仅训 adapter+头; 名含 finetune 实为 frozen)
  · source_mmd_physics: S2 冻结 → S3 **全量微调 + MMD 对齐 + ρ·L_phys 物理一致性** (T13: ρ·MSE(hi_pred, damage_norm) Arrhenius+CM 损伤监督)
  · random_frozen: 随机 encoder + 冻结 (P0-3 初始化对照, 判源负迁移归属)
  · ch_* (M7 通道级): ChannelSeqDataset + canonical 4 维; level=channel
  · cross_level_transfer (M7 层级消融): 旧服务级口径 (build_array_hi 12 维); level=service
