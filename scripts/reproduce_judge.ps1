# =====================================================================
# reproduce_judge (PowerShell 包装) — 与 reproduce_judge.sh 同参数/退出码/输出目录语义
#   用法: pwsh scripts/reproduce_judge.ps1 [-Output outputs/judge] [-Config configs/phased_array.yaml]
#   退出码: 0=成功 (REPRODUCE_OK 已写出); 非零=任一主步骤失败
# =====================================================================
param(
    [string]$Output = "outputs/judge",
    [string]$Config = "configs/phased_array.yaml"
)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$Python = if ($env:PYTHON) { $env:PYTHON } else { "python" }
& $Python scripts/reproduce_judge.py --output $Output --config $Config
exit $LASTEXITCODE
