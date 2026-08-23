# F1-A 终审修复清单（3 Important + 2 Minor）

终审（opus）无 Critical，但下列项合前修。组件仓根即本仓库根，分支 feature/f1a-closeout-rul-v2（不要切分支）。Python 为本机 conda `pytorch_gpu` 环境解释器（绝对路径已按公开仓扫描规则匿名化，2026-08-23）。

## I-1（Important）scripts/run_fault_model_eval.py 评估侧 rul_norm 硬编码 4088

问题：`build_windows` 对 v2 h5 已正确用 H=11688（读 h5 meta），但 `eval_rmse_phm(model, x, rul, event, rul_norm=4088.0)`（line 128）默认值仍是 4088，而调用方（173/184）都没传 rul_norm，于是 pred/true 各乘 4088 反归一 → 绝对 RMSE 比真实窗口数小 0.35 倍、PHM clip(±500) 也在错尺度。

修法（让两边用同一个 H，禁止默认值漂移）：
1. `build_windows` 解析出的 `rul_norm` 随返回值带出：把 `return x_arr, rs_arr, evs_arr` 改为 `return x_arr, rs_arr, evs_arr, rul_norm_float`（函数内已算出 rul_norm/rul_field，确保 v1 路径也有确定值 4088.0、v2 路径是 H）。
2. 所有 `build_windows(...)` 调用点（约 line 163/164/172/183）改成接收 4 元组 `x, r, ev, rn = ...`。
3. `eval_rmse_phm(..., rul_norm=4088.0)` 把默认值去掉改为必填位置/关键字参数（`def eval_rmse_phm(model, x, rul, event, rul_norm)`），两个调用点（173/184）把从 build_windows 拿到的 rn 传进去。
4. 验证：v2 下标称/故障两边都用同一 H，rmse_ratio 不受影响但绝对 RMSE 回到正确窗口尺度。跑该脚本的导入/语法检查（不一定要跑全脚本，故障 h5 可能不在盘上）。

## I-2（Important）scripts/calibrate_channel_threshold.py 读已删除 config 键

line 92 `cap_ratio = float(ch["rul_cap_ratio"])`：F1-A 步骤6 删了 `channel_level.rul_cap_ratio`，脚本一跑即 KeyError。该脚本调用的是 v1 `build_channel_labels`（确实需要 cap_ratio 参数），所以保留 0.35 语义：改为 `cap_ratio = float(ch.get("rul_cap_ratio", 0.35))`。

## I-3（Important）src/experiments/run_groups.py channel v1 分支 factor 错

当前 channel 分支对 v1 数据调 `_resolve_rul_scale(level, ch_cfg, tc)`，而该函数按 **config policy**（mission_horizon）返回 (1.0, H)，不看数据 schema——若默认 config 指向 v1 h5（窗口数标签，需 /4088），会拿 factor=1.0 让未归一标签进模型。注释写的是 "v1 legacy: factor=rul_max_norm" 但代码相反。

修法：channel 分支的 else（非 v2）直接用 service 上限，不调 `_resolve_rul_scale`：
```python
else:
    # channel v1 (legacy): rul_ch 为窗口数, 按 transfer.rul_max_norm(4088) 归一
    rul_factor = float(tc["rul_max_norm"])
    rul_scale_windows = rul_factor
```
保留 `_resolve_rul_scale` 函数（service fallback / 测试仍引用 test_rul_scale_policy）。

## Minor-1 src/experiments/run_groups.py service 回退分支对 wheel config 会 KeyError

line ~591 else 分支 `_resolve_rul_scale(level, cfg["channel_level"], tc)` 用 `cfg["channel_level"]`，但飞轮 wheel config 无 channel_level 段。改为 `cfg.get("channel_level", {})`（_resolve_rul_scale 对非 channel 层级本不读它，只是取个默认）。

## Minor-3 docs/DATA_DICTIONARY.md:87 旧口径

`| 通道级 rul_ch | 自当前窗到 z≥1 的窗数，封顶 0.35T；失效后截断（P0-1） |` 未反映 v2 双字段。改为说明：
- v2（channel_label_v2，默认）：写 `rul_ch_windows`（到 z≥1 的绝对窗口数，失效后截断/不封顶）+ `rul_ch_norm`（=rul_ch_windows/H，H=rul_scale_windows=11688）；不再写旧 `rul_ch`。
- v1（legacy）：`rul_ch` 封顶 0.35T（仅旧数据兼容）。
改这一行表格即可，顺带检查该文件有无别处提到 rul_ch 需同步。

## 约束
- 只改上述 4 个文件（run_fault_model_eval.py / calibrate_channel_threshold.py / run_groups.py / DATA_DICTIONARY.md）。不改其他、不改 src 其他生产逻辑。
- 改完跑：`/f/anaconda3/envs/pytorch_gpu/python.exe -m pytest tests/test_channel_hi.py tests/test_channel_baselines.py tests/test_models_factory.py tests/test_rul_scale_policy.py tests/test_channel_seq.py -q`（应仍全绿），并对 run_fault_model_eval/calibrate 做 `python -c "import ast; ast.parse(...)"` 或 import 检查。
- 不 commit（controller 审查后统一提交）。不派子 agent。

## 报告
写 `.superpowers/sdd/f1a-closeout/final-fixes-report.md`：每处改了什么、验证命令与结果、任何疑虑。返回只给状态、改动文件、测试计数一行。
