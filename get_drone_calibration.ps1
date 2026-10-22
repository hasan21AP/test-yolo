$src = "D:\AI\test-yolo\drone_dataset_source\val"
$dst = "D:\AI\test-yolo\drone_calibration_verified"

New-Item -ItemType Directory -Force $dst | Out-Null

Get-ChildItem $src -File -Filter "*.jpg" | Where-Object {
    $label = Join-Path $src ($_.BaseName + ".txt")
    (Test-Path $label) -and ((Get-Item $label).Length -gt 0)
} | Copy-Item -Destination $dst

(Get-ChildItem $dst -File).Count