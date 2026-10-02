# 日常训练统一入口；不指定类型时显示中文菜单。
[CmdletBinding()]
param(
    [ValidateSet('dense', 'mhc', 'moe', 'hybrid_moe', 'engram', 'ddp', 'moe_ddp', 'sft', 'ei', 'dpo', 'grpo', 'tron')]
    [string]$TrainType,
    [switch]$ListTypes,
    [switch]$SmokeTest,
    [switch]$DryRun,
    [switch]$UseWandb,
    [string]$OutputDir,
    [string]$ResumeFrom,
    [int]$TotalSteps,
    [int]$BatchSize,
    [int]$SeqLen,
    [int]$DModel,
    [int]$NHead,
    [int]$NLayer,
    [int]$DFF,
    [int]$VocabSize,
    [int]$NExperts,
    [int]$TopK,
    [int]$Epochs,
    [int]$Rounds,
    [int]$NProcesses,
    [int]$GradientAccumulationSteps,
    [double]$LearningRate,
    [ValidateSet('auto', 'cpu', 'cuda')]
    [string]$Device = 'auto',
    [ValidateSet('auto', 'float32', 'float16', 'bfloat16')]
    [string]$DType = 'auto',
    [ValidateSet('auto', 'gloo', 'nccl')]
    [string]$Backend = 'auto',
    [ValidateSet('uint16', 'uint32', 'int64')]
    [string]$DataDType = 'int64',
    [string]$TrainDataPath,
    [string]$ValidDataPath,
    [string]$TokenizerPath,
    [string]$ModelId,
    [string]$CheckpointPath,
    [string]$PromptPath,
    [string]$ConfigPath,
    [string[]]$ExtraArgs
)

$ErrorActionPreference = 'Stop'
$previousEncoding = $env:PYTHONIOENCODING
Push-Location $PSScriptRoot
try {
    $pythonPath = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
    if (-not (Test-Path -LiteralPath $pythonPath)) {
        throw '缺少 .venv/Scripts/python.exe，请先安装项目训练环境'
    }
    $env:PYTHONIOENCODING = 'utf-8'
    $trainingArguments = @('-m', 'utils.training_launcher')
    $parameterMap = @{
        TrainType = 'train-type'; OutputDir = 'output-dir'; ResumeFrom = 'resume-from'
        TotalSteps = 'total-steps'; BatchSize = 'batch-size'; SeqLen = 'seq-len'
        DModel = 'd-model'; NHead = 'n-head'; NLayer = 'n-layer'; DFF = 'd-ff'
        VocabSize = 'vocab-size'; NExperts = 'n-experts'; TopK = 'top-k'
        Epochs = 'epochs'; Rounds = 'rounds'; NProcesses = 'n-processes'
        GradientAccumulationSteps = 'gradient-accumulation-steps'; LearningRate = 'learning-rate'
        Device = 'device'; DType = 'dtype'; Backend = 'backend'; DataDType = 'data-dtype'
        TrainDataPath = 'train-data-path'; ValidDataPath = 'valid-data-path'
        TokenizerPath = 'tokenizer-path'; ModelId = 'model-id'; CheckpointPath = 'checkpoint-path'
        PromptPath = 'prompt-path'; ConfigPath = 'config-path'
    }
    # 只传入显式设置的参数，让各训练方式使用自身的默认值。
    foreach ($name in $parameterMap.Keys) {
        if ($PSBoundParameters.ContainsKey($name)) {
            $value = $PSBoundParameters[$name]
            if ($value -is [double]) {
                $value = $value.ToString([System.Globalization.CultureInfo]::InvariantCulture)
            }
            $trainingArguments += @(('--' + $parameterMap[$name]), [string]$value)
        }
    }
    if ($ListTypes) { $trainingArguments += '--list-types' }
    if ($SmokeTest) { $trainingArguments += '--smoke-test' }
    if ($DryRun) { $trainingArguments += '--dry-run' }
    if ($UseWandb) { $trainingArguments += '--use-wandb' }
    if ($ExtraArgs) { $trainingArguments += @('--') + $ExtraArgs }
    & $pythonPath @trainingArguments
    if ($LASTEXITCODE -ne 0) { throw "训练入口退出，退出码：$LASTEXITCODE" }
} finally {
    $env:PYTHONIOENCODING = $previousEncoding
    Pop-Location
}
