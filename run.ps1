$ErrorActionPreference = "Stop"
$env:UV_LINK_MODE = "copy"
$env:JUPYTER_CONFIG_DIR = Join-Path $PSScriptRoot ".jupyter_config"
New-Item -ItemType Directory -Force -Path $env:JUPYTER_CONFIG_DIR | Out-Null

Set-Location -LiteralPath $PSScriptRoot

uv sync
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

uv run python build_notebook.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

uv run jupyter nbconvert --to notebook --execute optimal_search.ipynb --inplace --ExecutePreprocessor.timeout=900
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

uv run jupyter nbconvert --to html optimal_search.ipynb --output optimal_search.html
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

uv run pytest -q
exit $LASTEXITCODE

