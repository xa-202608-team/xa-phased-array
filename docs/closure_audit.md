# 工程收口审计记录（F0）

> 2026-08-21 · 组件仓 · 冻结基线 `c7b7a8e`（main）
> 对应父仓 `transfer_clear_review_and_positive_gain_plan.md` §4g 终局 + 评审裁定收口清单。

## F0-1 冻结基线与分支状态

| 项 | 值 |
|---|---|
| 冻结提交（main） | `c7b7a8e` promotion: v0.3.0 候选（迁移终局 + CI 契约验证修复） |
| 迁移终局内容 | 清零重审 / k-shot 源域臂 / A1 α-soft / A3 layer-wise / B sim_v1→sim_v2 / §4h 偏差诊断（PR #11 → dev，PR #12 → main） |
| 计算区 checkout | `6fa9c07` @ `fix/migration-experiment-clear`（本分支 HEAD），与 main 内容一致（squash 合并后 main = 分支全量） |

## F0-2 stash `bd16c3d` 审计

**结论：归档证据后丢弃。** 不 restore、不合并、不用于任何计算。

### 内容

对 base `ff4f189`（清零重审 commit，2026-08-17 09:54）的**反向修改快照**，撤销三项 P0 修复：

| 文件 | stash 改法 | 与现役契约（origin/main）对比 |
|---|---|---|
| `build_channel_hi.py` | `T_dev = sa_feat_s[:,SA_COL_TJ]`（删 `-273.15`） | main:`- 273.15` 保留（修复正确）。stash 回退为 Kelvin 错误 |
| `run_groups.py` `_GROUP_MAP` | 迁移归因组 encoder 改回 `None`/`tcn`（破坏架构统一） | main: 迁移归因组统一显式 `gru` + 注释说明架构纪律 |
| `run_groups.py` `damageT` | `np.zeros_like(rulT)`（L_phys 全零占位） | main: `None`（通道级 L_phys 条件禁用，P0-1 修复） |
| `INPUT_SCHEMA.md` / `DATA_DICTIONARY.md` | 删"仿真器输出 Kelvin 已转 °C"说明 | main: 文档保留正确说明 |

### 判定

- stash 是**清零重审 P0 修复的撤销快照**（温度单位、架构纪律、L_phys 污染三处全部回退）；
- 该 stash 只存在于 `fix/migration-experiment-clear` 分支中途，base 不在 main 线；
- 最终分支经 PR #11（squash `0f4d8d7`）→ PR #12（squash `c7b7a8e`）完整合并进 main，**main 上为最终正确状态**；
- 因此 stash 是"分支开发过程的中间残留快照"，不构成对 main 的威胁，也无合并价值。

### 处置

- 2026-08-21 由评审裁定"先审查内容再决定归档/丢弃"；
- 本文件即**归档证据**（diff 全文见 git：`stash@{0}` 已聚焦于此记录）；
- 经用户确认后执行 `git stash drop stash@{0}`（见 F0 汇报）。