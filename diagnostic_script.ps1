# Banana Ripeness Detection - Safe Model Diagnostic
# READ-ONLY: Does not modify, rename, move, delete, or overwrite models.

$ProjectDir = "D:\Pranjal Files\Capstone2026\Banana-Ripeness-Bunch"
$ModelsDir = Join-Path $ProjectDir "Models"

Write-Host "==============================================="
Write-Host " BANANA MODEL DIAGNOSTIC"
Write-Host "==============================================="
Write-Host ""

if (-not (Test-Path $ProjectDir)) {
    Write-Host "ERROR: Project directory not found:"
    Write-Host $ProjectDir
    exit 1
}

Set-Location $ProjectDir

Write-Host "PROJECT:"
Write-Host $ProjectDir
Write-Host ""

Write-Host "==============================================="
Write-Host " MODEL FILES"
Write-Host "==============================================="

$models = @(
    "best.pt",
    "banana_detector.pt",
    "banana_stage_mobilenet_final.keras"
)

foreach ($model in $models) {

    $path = Join-Path $ModelsDir $model

    if (Test-Path $path) {

        $file = Get-Item $path

        Write-Host ""
        Write-Host "FOUND: $model"
        Write-Host "Full Path: $($file.FullName)"
        Write-Host "Size MB: $([math]::Round($file.Length / 1MB, 2))"
        Write-Host "Modified: $($file.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'))"

    } else {

        Write-Host ""
        Write-Host "NOT FOUND: $model"
        Write-Host "Expected: $path"
    }
}

Write-Host ""
Write-Host "==============================================="
Write-Host " PYTHON ENVIRONMENT"
Write-Host "==============================================="

$python = Get-Command python -ErrorAction SilentlyContinue

if ($python) {

    Write-Host "Python found:"
    python --version

} else {

    Write-Host "Python NOT FOUND in PATH."
    Write-Host "Install/use your project's Python environment before continuing."
    exit 0
}

Write-Host ""
Write-Host "==============================================="
Write-Host " PYTORCH CHECK"
Write-Host "==============================================="

$torchCheck = python -c "import torch; print(torch.__version__)" 2>&1

if ($LASTEXITCODE -eq 0) {

    Write-Host "PyTorch installed:"
    Write-Host $torchCheck

} else {

    Write-Host "PyTorch is NOT available."
    Write-Host ""
    Write-Host "Potential installation command:"
    Write-Host "pip install torch torchvision"
    Write-Host ""
    Write-Host "STOPPING BEFORE MODEL INSPECTION."
    exit 0
}

Write-Host ""
Write-Host "==============================================="
Write-Host " ULTRALYTICS CHECK"
Write-Host "==============================================="

$yoloCheck = python -c "from ultralytics import YOLO; print('Ultralytics available')" 2>&1

if ($LASTEXITCODE -eq 0) {

    Write-Host $yoloCheck

} else {

    Write-Host "Ultralytics is NOT available."
    Write-Host ""
    Write-Host "Potential installation command:"
    Write-Host "pip install ultralytics"
    Write-Host ""
    Write-Host "STOPPING BEFORE best.pt INSPECTION."
    exit 0
}

$BestModel = Join-Path $ModelsDir "best.pt"

if (-not (Test-Path $BestModel)) {

    Write-Host ""
    Write-Host "best.pt was not found."
    exit 0
}

Write-Host ""
Write-Host "==============================================="
Write-Host " INSPECTING best.pt"
Write-Host "==============================================="

$pythonScript = @'
from ultralytics import YOLO
import os

path = r"""BEST_MODEL_PATH"""

print("Loading:", path)
print()

try:
    model = YOLO(path)

    print("MODEL LOADED SUCCESSFULLY")
    print()

    print("Task:")
    print(getattr(model, "task", "UNKNOWN"))
    print()

    print("Model type:")
    print(type(model.model))
    print()

    print("Class names:")
    names = getattr(model, "names", None)

    if names is not None:
        print(names)
    else:
        print("UNKNOWN")

    print()

    if names is not None:
        try:
            print("Number of classes:")
            print(len(names))
        except:
            print("Could not determine number of classes.")

    print()

    print("Model configuration:")
    try:
        print(model.model.yaml)
    except Exception as e:
        print("Could not read model configuration:", e)

    print()

    print("Model info:")
    try:
        model.info(verbose=True)
    except Exception as e:
        print("Could not generate model info:", e)

except Exception as e:

    print("FAILED TO LOAD best.pt")
    print()
    print(type(e).__name__)
    print(str(e))

'@

$pythonScript = $pythonScript.Replace(
    "BEST_MODEL_PATH",
    $BestModel.Replace("\", "\\")
)

$tempScript = Join-Path $env:TEMP "banana_best_pt_inspection.py"

Set-Content -Path $tempScript -Value $pythonScript -Encoding UTF8

python $tempScript

Write-Host ""
Write-Host "==============================================="
Write-Host " DIAGNOSTIC COMPLETE"
Write-Host "==============================================="
Write-Host ""
Write-Host "No model files were modified."
Write-Host "No training was performed."
Write-Host "No application files were modified."
Write-Host ""
Write-Host "Copy the COMPLETE output and send it back."