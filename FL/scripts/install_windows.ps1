# Windows + Python 3.11: install TensorFlow stack, then TFF without pulling grpcio 1.46 sdist.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host ""
Write-Host "If pip warns about 'invalid distribution ~ip', delete matching folders under:"
Write-Host "  Python311\Lib\site-packages  (names starting with ~ip)"
Write-Host ""

Write-Host "Step 0: PyYAML 6.x wheel (avoid 5.4.1 sdist build failure on Python 3.11)..."
pip install --no-warn-conflicts "PyYAML>=6.0.1,<7"

Write-Host "Step 1: core requirements (TensorFlow + jax + grpcio wheels)..."
pip install --no-warn-conflicts -r requirements.txt

Write-Host "Step 2: tensorflow-privacy (no deps; dp-accounting comes from requirements.txt)..."
pip install --no-warn-conflicts tensorflow-privacy==0.8.10 --no-deps

Write-Host "Step 3: TensorFlow Federated (ignore declared grpcio pin; use wheels from step 1)..."
pip install --no-warn-conflicts tensorflow-federated==0.33.0 --no-deps

Write-Host "Verify..."
python -c "import tensorflow as tf; import tensorflow_federated as tff; print('ok tensorflow', tf.__version__, 'tff', tff.__version__)"
