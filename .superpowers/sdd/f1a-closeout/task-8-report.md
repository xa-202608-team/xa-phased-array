# F1-A 任务 8 报告：更新旧测试 + 新增无大数据 HDF5 fixture

状态：**测试实现完成，但被一个一行的生产代码 bug 阻塞（未改 src，供审查者裁决）**

## 一、改动文件清单（工作树，未 git add / commit）

| 文件 | 改动类型 | 说明 |
|------|---------|------|
| `tests/test_channel_hi.py` | 新增 8 条 fixture 测试 + 修正 2 条陈旧 v2 单元测试 | 8.1 最小 v2 HDF5 fixture（tmp_path，无大数据）；陈旧断言改为匹配已提交的"函数返回绝对窗口数"语义 |
| `tests/test_channel_baselines.py` | docstring + 断言修正 + 新增值域断言 | 8.2 v2 H / v1 4088 双口径；rul_max_norm 改从 h5 meta 读；`_load_channels` 的 `rul_ch` key 在 v2 下值域断言为窗口数 |
| `tests/test_models_factory.py` | docstring + 新增 1 条 config 驱动测试 | 8.3 factory 从 config.model.gru 显式读取验证（deepcopy 改 128/1） |
| `tests/test_rul_scale_policy.py` | 注释更新 | 8.4 标注 channel+mission_horizon 为 legacy/v1 兼容回归路径，v2 生产走 h5 meta |
| （非我改动，已存在）`scripts/plot_trajectory.py` | 工作树预先已修改 | 任务开始前即为 `M` 状态，本次未触碰 |

## 二、新增测试名与条数

**8.1（test_channel_hi.py，共 8 条，全部 tmp_path 无大数据）：**
- `test_read_channel_label_meta_v2_ok` — 合法 v2 h5 的 meta 强校验字段
- `test_v2_fixture_dual_fields_no_legacy_rul_ch` — 双字段存在 + `rul_ch_norm*H==rul_ch_windows` + 读旧 `rul_ch` 触发 KeyError
- `test_load_target_channel_v2_fixture` — loader 返回 `rul==rul_ch_norm`（已归一），shape 一致，覆盖失效/删失双通道
- `test_meta_missing_schema_attr_valueerror` — 缺 `channel_label_schema` → ValueError
- `test_meta_unknown_schema_valueerror` — 未知 schema → ValueError
- `test_meta_v2_missing_rul_scale_windows_valueerror` — v2 缺 `rul_scale_windows` → ValueError
- `test_meta_v2_non_degc_t_dev_unit_valueerror` — `t_dev_unit!='degC'` → ValueError
- `test_meta_v2_rul_capped_true_valueerror` — `rul_capped!='false'` → ValueError

**8.3（test_models_factory.py，共 1 条）：**
- `test_factory_reads_gru_from_config_deepcopy` — deepcopy 改 `model.gru.hidden=128/num_layers=1`，断言模型 encoder.gru 跟随（证明读 config 而非硬编码）

**合计新增 9 条。** 另有 2 条存量 v2 单元测试被修正断言、3 处 docstring/断言按 brief 更新（见 self-review）。

## 三、pytest 完整输出（4 文件，当前工作树，未打生产补丁）

```
=========================== short test summary info ===========================
FAILED tests/test_channel_hi.py::test_read_channel_label_meta_v2_ok - NameErr...
FAILED tests/test_channel_hi.py::test_load_target_channel_v2_fixture - NameEr...
FAILED tests/test_channel_hi.py::test_meta_v2_missing_rul_scale_windows_valueerror
FAILED tests/test_channel_hi.py::test_meta_v2_non_degc_t_dev_unit_valueerror
FAILED tests/test_channel_hi.py::test_meta_v2_rul_capped_true_valueerror - Na...
FAILED tests/test_channel_baselines.py::test_load_channels_structure - NameEr...
FAILED tests/test_channel_baselines.py::test_evaluate_channel_baselines_full
FAILED tests/test_channel_baselines.py::test_split_by_trajectory_consistent
FAILED tests/test_channel_baselines.py::test_evaluate_reproducible_same_seed
FAILED tests/test_channel_baselines.py::test_similarity_better_than_constant_typically
======================== 10 failed, 36 passed in 2.53s =========================
```

**passed/failed/skipped：36 passed / 10 failed / 0 skipped。**

逐文件：`test_channel_hi.py` 14 passed + 5 failed；`test_channel_baselines.py` 14 passed + 5 failed；`test_models_factory.py` 5 passed；`test_rul_scale_policy.py` 3 passed。

**全部 10 个失败同根：`NameError: name '_CHANNEL_META_REQUIRED' is not defined`（生产代码，见"生产疑虑 1"）。** 它们各自都经过 `read_channel_label_meta`（或经 `_load_channels`/`evaluate_channel_baselines` 间接）。

**关键验证**：我用"运行时 monkeypatch（不改任何文件）"把 `channel_dataset._CHANNEL_META_REQUIRED = channel_dataset._CHANNEL_META_V2_REQUIRED` 后，8.1 全部 fixture 测试逻辑逐条跑通（PASS ×8），证明**测试本身正确，只被这一行生产 bug 阻塞**。该 monkeypatch 脚本未写入仓库、未改动 src。

## 四、self-review（自行发现并解决的问题）

1. **我的 helper 初版 bug**：`_write_min_v2_h5` 里 `_rul_pair` 在 `rul_scale_windows=None` 负例下 `float(None)` 崩。已修复：None 时直接填 rul_w（负例内容本就不被强校验读取）。
2. **两条存量测试陈旧（已修正）**：`test_channel_label_v2_failed_rul_absolute_no_cap` 与 `test_channel_label_v2_censored_lb_scale_by_h` 在 cc8f214 起**已红**（提交前即红，非我引入）。原因：f5e1df0 时 `build_channel_labels_v2` 返回归一值 `(eol-t)/H`，cc8f214（F1-A 步骤 1-7）改为返回**绝对窗口数**（docstring 明示"写盘时再除 H 得 rul_ch_norm"），但这两条测试未同步更新，仍断言归一语义。我按已提交语义修正为断言绝对窗口数（`rul[:eol]==eol-t`、`rul[0]==eol` 等），归一验证移到写盘/loader 边界（由我新增的 dual-fields / load_target_channel fixture 断言承接）。
3. **8.3 新增测试隔离**：用 `copy.deepcopy(_cfg())` 改配置后再构建，避免污染共享 cfg（与 `_smoke_cfg` 同模式）。
4. **8.4 定位确认**：`_resolve_rul_scale` 对 channel+mission_horizon 在 v2 生产路径确实无调用方（run_groups channel 分支直接从 h5 meta 读 factor=1.0/H，`run_groups.py:574-579`）；函数仍被 service fallback 与 v1 legacy 使用，故按 brief 保留函数/测试，仅加注释。

## 五、生产代码疑虑（未自行修改）

1. **【阻塞级，须一行修复】** `src/transfer/channel_dataset.py:63`：
   `for k in _CHANNEL_META_REQUIRED:` — `_CHANNEL_META_REQUIRED` 未定义，模块内只定义了 `_CHANNEL_META_V2_REQUIRED`（第 38 行）。→ **所有 v2 channel_features.h5 读数路径一读即 `NameError`**：`read_channel_label_meta`、`load_target_channel`、`channel_baselines._load_channels`、`evaluate_channel_baselines`，直接阻塞通道级 run_groups 与基线。
   建议最小修复（一行，纯改名，非逻辑改动）：
   ```python
   for k in _CHANNEL_META_V2_REQUIRED:
   ```
   修后 8.1 的 5 条 fixture 测试 + 基线的 5 条 e2e 测试应转绿（后者还取决于数据重建，见 2）。

2. **【数据工件，非测试问题】** 本机 `data/features/phased_array/schema_ch_v1/target/channel_features.h5`（680MB）是**混合态**：顶层 attrs 已是 v2（`channel_label_schema=channel_label_v2`、`rul_scale_windows=11688`），但子组只有旧 `rul_ch` 单字段（已归一值），**没有** `rul_ch_windows`/`rul_ch_norm`。即使修了问题 1，`_load_channels`/`load_target_channel` 读 v2 字段仍会 `KeyError`。需重建：`python -m src.sim.build_channel_hi --report`。**clean checkout（无大数据）时这些 e2e 测试本就 `skip`，故不影响 clean-checkout 的回归面**；我的 8.1 fixture 测试不依赖该文件、总是执行。

3. **8.4 附带确认**：`_resolve_rul_scale(level="channel", policy=mission_horizon)` 已无 v2 生产调用方，但函数未删（service/v1 fallback 仍用），按 brief 未删函数/测试，仅注释。

4. 次要：存量测试里 `zoo = np.zeros(T)` 为未用变量（历史遗留，无害）；本次未动。

## 六、结论

8.1–8.4 的**测试代码全部完成且自检通过**；覆盖"clean checkout 无见证"的最小 v2 HDF5 fixture 测试就绪。当前 pytest 输出 36 passed / 10 failed / 0 skipped，10 个失败全部由上述第 1 条**一行生产 typo** 导致（已用运行时 monkeypatch 证明：补丁后 8.1 全部 fixture 逻辑 PASS）。审查者批准该一行修复后，本批测试即达全绿（本地 e2e 另需重建数据）。