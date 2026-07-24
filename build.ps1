# Build both WASM cores of the merged Squirrel Engine and stage them into web/.
#
#   physics core -> web/pkg/squirrel_physics*      (Rapier sim)
#   render core  -> web/engine/squirrel_engine*    (WebGPU renderer)
#
# web/engine/renderer.js is the hand-written JS renderer and is left untouched:
# the render core is built into its own crate-local pkg/ and only the generated
# squirrel_engine* files are copied across, so wasm-pack never clears it.
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

Write-Host "Building squirrel-physics (WASM)..." -ForegroundColor Cyan
Push-Location (Join-Path $root "physics")
try { wasm-pack build --target web --out-dir ../web/pkg --release } finally { Pop-Location }

Write-Host "Building squirrel-engine renderer (WASM)..." -ForegroundColor Cyan
Push-Location (Join-Path $root "render")
try { wasm-pack build --target web --release } finally { Pop-Location }

Write-Host "Staging render core into web/engine..." -ForegroundColor Cyan
$dstEngine = Join-Path $root "web/engine"
New-Item -ItemType Directory -Force -Path $dstEngine | Out-Null
Copy-Item (Join-Path $root "render/pkg/squirrel_engine*.*") $dstEngine -Force

Write-Host ""
Write-Host "Build complete. To run:" -ForegroundColor Green
Write-Host "    node serve.mjs"
Write-Host "    open http://localhost:8083  (needs a WebGPU browser: Chrome/Edge 113+)"
