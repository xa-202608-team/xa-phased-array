# =====================================================================
# reproduce_full (PowerShell 包装) — 与 reproduce_full.sh 同参数/退出码/输出目录语义
#   用法: pwsh scripts/reproduce_full.ps1 [-Output outputs/full] [-Config cfg.yaml]
#         [-NTraj 200] [-Seeds 5] [-Fast]
#   退出码: 0=成功 (REPRODUCE_OK 已写出); 非零=任一主步骤失败 (不吞错)
# =====================================================================
param(
    [string]$Output = "outputs/full",
    [string]$Config = "configs/phased_array.yaml",
    [int]$NTraj = 200,
    [int]$Seeds = 5,
    [switch]$Fast
)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$Python = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$Args = @("scripts/reproduce_full.py", "--output", $Output, "--config", $Config,
          "--n-traj", "$NTraj", "--seeds", "$Seeds")
if ($Fast) { $Args += "--fast" }
& $Python @Args
exit $LASTEXITCODE
