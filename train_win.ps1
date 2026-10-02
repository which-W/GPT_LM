# 使用项目虚拟环境训练，并将额外参数传给训练程序。
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    & ./.venv/Scripts/python.exe -m train --train_data_path data/TinyStories-train.bin --valid_data_path data/TinyStories-valid.bin --data_dtype int64 @args
    if ($LASTEXITCODE -ne 0) { throw "训练失败，退出码：$LASTEXITCODE" }
} finally {
    Pop-Location
}
