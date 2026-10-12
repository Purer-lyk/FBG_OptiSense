param(
    [string]$BuildName = 'customer_beta_1_0_8_20261012'
)

$ErrorActionPreference = 'Stop'
if ($BuildName -notmatch '^[a-zA-Z0-9_-]+$') {
    throw 'BuildName must be a single safe directory name'
}
$deployment = Split-Path -Parent $MyInvocation.MyCommand.Path
$repository = (Resolve-Path (Join-Path $deployment '..')).Path
$sourceVenv = (Resolve-Path (Join-Path $repository 'Python\.venv')).Path
$spec = Join-Path $deployment 'pyinstaller\fbg_optisense_studio_beta.spec'
$buildRoot = Join-Path $repository ("build\$BuildName")
$distRoot = Join-Path $buildRoot 'dist'
$clientRoot = Join-Path $distRoot 'app_beta_build'
if (Test-Path -LiteralPath $clientRoot) {
    throw "Beta output already exists; choose a new BuildName: $clientRoot"
}

# A release-scoped ASCII alias avoids Qt-hook issues with Chinese venv paths.
# This does not change PATH, global configuration or the source environment.
$asciiBuildRoot = Join-Path $env:LOCALAPPDATA ("FBGOptiSenseBuild\$BuildName")
$asciiVenv = Join-Path $asciiBuildRoot 'venv'
if (-not (Test-Path -LiteralPath $asciiBuildRoot)) {
    New-Item -ItemType Directory -Path $asciiBuildRoot | Out-Null
}
if (Test-Path -LiteralPath $asciiVenv) {
    $junction = Get-Item -LiteralPath $asciiVenv
    if ($junction.LinkType -ne 'Junction' -or [string]$junction.Target -ne $sourceVenv) {
        throw "Existing alias is not this source venv: $asciiVenv"
    }
} else {
    New-Item -ItemType Junction -Path $asciiVenv -Target $sourceVenv | Out-Null
}
$asciiPython = Join-Path $asciiVenv 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $asciiPython)) { throw "Missing build Python: $asciiPython" }
Push-Location $repository
try {
    & $asciiPython -m PyInstaller --noconfirm --clean `
        --workpath (Join-Path $buildRoot 'pyinstaller') `
        --distpath $distRoot $spec
    if ($LASTEXITCODE -ne 0) { throw 'Beta client PyInstaller build failed' }
} finally {
    Pop-Location
}
if (-not (Test-Path -LiteralPath (Join-Path $clientRoot 'FBG_OptiSense_Studio.exe'))) {
    throw 'Expected frozen Beta client is missing'
}
Write-Host "Beta client: $clientRoot"
Write-Host 'This builds only a client payload: no installer, updater signing or hardware operation.'
