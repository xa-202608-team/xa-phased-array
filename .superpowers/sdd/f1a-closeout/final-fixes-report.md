# F1-A 终审修复报告（final-fixes）

日期：2026-08-22　分支：feature/f1a-closeout-rul-v2（未切分支、未 commit）
Python：`/f/anaconda3/envs/pytorch_gpu/python.exe`

## 改动逐项

### I-1（Important）`scripts/run_fault_model_eval.py` — 评估侧 rul_norm 硬编码 4088

- `build_windows` 返回 4 元组：把 `return x_arr, rs_arr, evs_arr` 改为
  `return x_arr, np.array(rs, ...), np.array(evs, ...), float(rul_norm)`；docstring 同步说明
  第 4 个返回值为归一因子（v2=H / v1=4088），供评估反归一、禁止评估侧默认值漂移。
- `eval_rmse_phm(model, x, rul, event, rul_norm=4088.0)` → 去掉默认值改为**必填**参数
  `def eval_rmse_phm(model, x, rul, event, rul_norm)`，docstring 注明须与 build_windows 同一因子。
- 全部调用点接收 4 元组：`x_tr/r_tr/ev_tr/_rn_tr`、`x_va/r_va/ev_va/_rn_va`、
  `x_nom/r_nom/ev_nom/rn_nom`、`x_f/r_f/ev_f/rn_f`；两个评估调用点分别传 `rn_nom` / `rn_f`。
  （train/val 解包的 `_rn_*` 仅作占位，训练用归一标签，不需要反归一。）
- 生效结果：v2 下标称/故障两侧用同一 H=11688（ich read from h5 meta），绝对 RMSE 回到
  真实窗口尺度，rmse_ratio 不受影响。

### I-2（Important）`scripts/calibrate_channel_threshold.py`

- line 92：`cap_ratio = float(ch["rul_cap_ratio"])` →
  `cap_ratio = float(ch.get("rul_cap_ratio", 0.35))`（并加注释：v2 已删该 config 键，
  脚本仍调 v1 `build_channel_labels` 需要 cap_ratio，故保留 0.35 语义作默认）。

### I-3（Important）`src/experiments/run_groups.py` — channel v1 分支 factor 错

- channel 分支 else（非 v2）改为按数据 schema 直接用服务上限，不再调 `_resolve_rul_scale`：
  ```python
  else:
      # channel v1 (legacy): rul_ch 为窗口数, 按 transfer.rul_max_norm(4088) 归一
      rul_factor = float(tc["rul_max_norm"])
      rul_scale_windows = rul_factor
  ```
- `_resolve_rul_scale` 函数保留未动（service fallback + `test_rul_scale_policy` 仍引用）。

### Minor-1 `src/experiments/run_groups.py` — service 回退对 wheel config KeyError

- service 回退分支 `cfg["channel_level"]` → `cfg.get("channel_level", {})`
  （wheel config 无 channel_level 段；`_resolve_rul_scale` 对非 channel 层级不读它，纯传参默认）。

### Minor-3 `docs/DATA_DICTIONARY.md:87` — 旧口径

- 单行 `rul_ch` 拆为两行：
  - v1 legacy（仅旧数据）：`rul_ch` 封顶 0.35T。
  - v2 默认：`rul_ch_windows`（到 z≥1 绝对窗口数，失效后截断不封顶）+ `rul_ch_norm`
    （=`rul_ch_windows`/H，H=`rul_scale_windows`=11688，h5 元数据携带），不再写旧 `rul_ch`。
- 已 grep 全文确认该文件仅此一处提及 `rul_ch`，无别处需同步。

## 验证

1. 语法检查：
   `/f/anaconda3/envs/pytorch_gpu/python.exe -c "import ast; ast.parse(open('scripts/run_fault_model_eval.py', encoding='utf-8').read()); ast.parse(open('scripts/calibrate_channel_threshold.py', encoding='utf-8').read()); ast.parse(open('src/experiments/run_groups.py', encoding='utf-8').read()); print('ast.parse OK: 3 files')"`
   → `ast.parse OK: 3 files`
2. Config 键核查（I-2/I-3 前置）：
   `transfer.rul_max_norm = 4088` 存在；`channel_level.rul_cap_ratio` 确实已删（False）。
3. brief 指定 pytest：
   `/f/anaconda3/envs/pytorch_gpu/python.exe -m pytest tests/test_channel_hi.py tests/test_channel_baselines.py tests/test_models_factory.py tests/test_rul_scale_policy.py tests/test_channel_seq.py -q`
   → **59 passed, 0 failed, 0 skipped**（788.69s / 13:08）。exit code 0。
4. 早期信号（最相关用例）：`tests/test_rul_scale_policy.py` 单独 3 passed in 1.77s
   （`_resolve_rul_scale` 完好，I-3/Minor-1 未破坏其引用）。

## 疑虑

- `.superpowers/sdd/f1a-closeout/progress.md` 有**非本次改动**的未提交变更
  （Task 8/9 complete、终审判定等——疑似 controller/他 agent 并发写入）。不在 brief 四个文件内，
  已保持原样未动，交由 controller 决定是否纳入收口提交。
- `eval_rmse_phm` 的 PHM clip 仍为固定 ±500（绝对窗口数）。brief 只要求传对 rul_norm；
  ±500 在 H=11688 尺度下相对更紧（在旧 4088 尺度已是同一固定窗口数）。如需「按比例 clip」
  应属新设计决策，未擅改。
- 本轮 pytest 走真实 h5 的 baselines 用例较重，单机约 13 分钟；与任务 9 全程 371 tests/925s
  量级一致，非本次改动引入。