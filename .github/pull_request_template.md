# Pull Request

## 关联 Issue

<!-- 必填：Closes #<编号>，或说明无对应 Issue 的原因 -->

## 变更范围

<!-- 必填：本次改动的目录/模块列表，如 configs/、src/train/ -->

## 明确不做内容

<!-- 必填：本次 PR 明确不涉及、留待后续的内容，防止范围蔓延 -->

## 数据/配置版本

<!-- 必填：涉及的数据集名称/版本/SHA256，配置文件版本或 config_sha256 -->

## 验收命令

<!-- 必填：在组件仓库根目录执行的验收命令及退出码，如 verify / reproduce --mode quick --output /outputs -->

## metrics.json

<!-- 必填：粘贴本次运行的 metrics.json 关键内容（或链接），须符合 schemas/metrics.schema.json -->

## manifest.json

<!-- 必填：粘贴本次运行的 manifest.json 关键内容（或链接），须符合 schemas/manifest.schema.json -->

## 风险

<!-- 必填：本次变更可能引入的风险（复现性、性能、兼容性等）及影响面 -->

## 回退方式

<!-- 必填：出问题时如何回退（revert commit / 版本号 / 数据回滚等） -->

## 组件契约版本

<!-- 必填：本次变更涉及/依赖的组件契约版本（如 component-contract-v1.0.0）；若改变契约兼容性须说明 -->

## 单指标影响

<!-- 必填：本次变更对核心指标（RUL RMSE / PHM / α-λ accuracy 等）的单一变量影响估计；纯工程/文档变更填"无指标影响" -->

## 是否需要 RC

<!-- 必填：本次变更是否触发 RC 打包（handoff/artifact-map.yaml 槽位工件变动 → 需要；仅代码/文档 → 不需要），并说明理由 -->

## 无标签泄漏声明

<!-- 必填：确认未把 HI/SOH/RUL 监督标签或其派生中间产物直接提交入库；数据依赖测试仅允许按"本地工件缺失"显式 skip -->

## 公开扫描结论

<!-- 必填：scan_public_repo --root . 的运行结论（退出码 + 违规计数）；任何 CRITICAL/HIGH 须在合并前清零 -->
