# build.ps1 —— DiffBlindWizard 一键打包(2026-11-07,PK)
# 版本号单一源:probes/gui_solver.py VERSION(报告/GUI 标题自动同步)
# 用法: powershell -ExecutionPolicy Bypass -File build.ps1
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Set-Location $root

$ver = (Select-String -Path "$root\probes\gui_solver.py" `
        -Pattern 'VERSION = "([^"]+)"').Matches[0].Groups[1].Value
Write-Host "== DiffBlindWizard v$ver 打包 ==" -ForegroundColor Cyan

# 1) 清旧产物
if (Test-Path "$root\dist\DiffBlindWizard") {
    Remove-Item "$root\dist\DiffBlindWizard" -Recurse -Force
}
if (Test-Path "$root\build\DiffBlindWizard") {
    Remove-Item "$root\build\DiffBlindWizard" -Recurse -Force
}

# 2) 打包(spec 含 pyFAI/fabio 数据与 hiddenimports;排除 torch)
python -m PyInstaller --noconfirm --clean DiffBlindWizard.spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller 失败 exit=$LASTEXITCODE" }

# 3) 校验
$exe = "$root\dist\DiffBlindWizard\DiffBlindWizard.exe"
if (-not (Test-Path $exe)) { throw "未找到 $exe" }
$mb = [math]::Round((Get-Item $exe).Length / 1MB, 1)
Write-Host "== 打包完成: dist\DiffBlindWizard ($($mb)MB 主程序) ==" -ForegroundColor Green

# 4) 桌面快捷方式(指向启动器,工作目录=dist\DiffBlindWizard)
$desk = [Environment]::GetFolderPath("Desktop")
$lnk  = Join-Path $desk "DiffBlindWizard v$ver.lnk"
$ws = New-Object -ComObject WScript.Shell
$sc = $ws.CreateShortcut($lnk)
$sc.TargetPath = $exe
$sc.WorkingDirectory = "$root\dist\DiffBlindWizard"
$sc.Description = "粉末衍射盲解向导 v$ver"
$sc.Save()
Write-Host "== 桌面快捷方式: $lnk ==" -ForegroundColor Green
