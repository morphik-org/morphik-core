<#
  Morphik Core one-liner installer + server launcher for Windows PowerShell.
  Mirrors install_and_start.sh (macOS/Linux) but uses Windows-friendly commands.

  Usage (PowerShell):
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
    ./install_and_start.ps1

  Requirements:
    - Docker Desktop running (used for local Redis container and optional services)
    - Python 3.10+ (3.12 recommended)
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Write-Info($msg) { Write-Host "`n[INFO] $msg" -ForegroundColor Cyan }
function Write-Step($msg) { Write-Host "`n[STEP] $msg" -ForegroundColor Yellow }
function Write-Ok($msg)   { Write-Host "`n[OK]   $msg" -ForegroundColor Green }
function Write-Err($msg)  { Write-Host "`n[ERR]  $msg" -ForegroundColor Red }

function Assert-Docker {
  if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Err "Docker is required (used to run a local Redis container). Install Docker Desktop first."
    throw "Docker not found"
  }
  cmd.exe /c "docker info >NUL 2>&1"
  if ($LASTEXITCODE -ne 0) {
    Write-Err "Docker is installed but not running. Please start Docker Desktop and re-run."
    throw "Docker not running"
  }
}

function Ensure-Uv {
  $uvCmd = Get-Command uv -ErrorAction SilentlyContinue
  if (-not $uvCmd) {
    Write-Step "Installing uv (Python packaging tool)..."
    if (Get-Command py -ErrorAction SilentlyContinue) {
      py -m pip install --user --upgrade uv
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
      python -m pip install --user --upgrade uv
    } else {
      Write-Err "Python not found. Please install Python 3.12 (recommended) from python.org, then re-run."
      throw "Python not found"
    }

    # Try to locate uv.exe in the user's Scripts directory if PATH wasn't updated
    $uvCmd = Get-Command uv -ErrorAction SilentlyContinue
    if (-not $uvCmd) {
      try {
        if (Get-Command py -ErrorAction SilentlyContinue) {
          $userBase = & py -c "import site; print(site.USER_BASE)" 2>$null
          $pyver = & py -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')" 2>$null
        } elseif (Get-Command python -ErrorAction SilentlyContinue) {
          $userBase = & python -c "import site; print(site.USER_BASE)" 2>$null
          $pyver = & python -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')" 2>$null
        } else { $userBase = $null }
      } catch { $userBase = $null }
      $candidates = @()
      if ($userBase) { $candidates += (Join-Path $userBase 'Scripts/uv.exe') }
      if ($pyver) {
        if ($env:APPDATA)      { $candidates += (Join-Path $env:APPDATA       ("Python/Python$pyver/Scripts/uv.exe")) }
        if ($env:LOCALAPPDATA) { $candidates += (Join-Path $env:LOCALAPPDATA  ("Programs/Python/Python$pyver/Scripts/uv.exe")) }
      }
      foreach ($cand in $candidates) {
        if ($cand -and (Test-Path $cand)) {
          $dir = [System.IO.Path]::GetDirectoryName($cand)
          if ($dir -and ($env:Path -notlike "*$dir*")) { $env:Path = "$dir;$env:Path" }
          $uvCmd = Get-Command uv -ErrorAction SilentlyContinue
          if ($uvCmd) { break }
        }
      }
    }

    if (-not $uvCmd) {
      Write-Err "uv was installed but not found on PATH. Open a new PowerShell window or add your Python Scripts folder to PATH, then retry."
      throw "uv not on PATH"
    }
  }
}

function Ensure-EnvFile {
  if (-not (Test-Path .env) -and (Test-Path .env.example)) {
    Write-Step "Creating default .env from .env.example..."
    Copy-Item .env.example .env
  }
}

function Ensure-Py312 {
  $exe = $null
  if (Get-Command py -ErrorAction SilentlyContinue) {
    try { $exe = & py -3.12 -c "import sys; print(sys.executable)" 2>$null } catch { $exe = $null }
  }
  if (-not $exe -or -not (Test-Path $exe)) {
    if (Get-Command python -ErrorAction SilentlyContinue) {
      try {
        $ver = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        if ($ver -eq '3.12') { $exe = (Get-Command python).Source }
      } catch { $exe = $null }
    }
  }
  if (-not $exe -or -not (Test-Path $exe)) {
    Write-Err "Python 3.12 not found. Please install Python 3.12 from python.org (or via the Microsoft Store), then re-run."
    throw "Python 3.12 not found"
  }
  Write-Info "Using Python 3.12 at: $exe"
  return $exe
}

$arch = $env:PROCESSOR_ARCHITECTURE
if (-not $arch) { $arch = "unknown" }
Write-Info "Detected OS: Windows | Arch: $arch"

# 1) Check Docker
Assert-Docker

# 2) Ensure uv is available
Ensure-Uv

# 3) Create/Sync virtual environment and install project deps
Write-Step "Installing project dependencies with uv..."
$py312 = Ensure-Py312
uv sync --python $py312

# 4) Ensure .env exists
Ensure-EnvFile

# 4b) Ensure we install into the project's venv (not a global Conda/Python)
$venvPython = Join-Path (Get-Location) '.venv\Scripts\python.exe'
if (-not (Test-Path $venvPython)) {
  Write-Err "Project virtual environment was not created at .venv. Please re-run 'uv sync'."
  throw ".venv not found"
}

# 5) Install ColPali engine (knowledge-graph/multimodal search)
Write-Step "Installing ColPali engine..."
uv pip install --python $venvPython `
  colpali-engine@git+https://github.com/illuin-tech/colpali@80fb72c9b827ecdb5687a3a8197077d0d01791b3

# 6) Install llama-cpp-python (CPU build by default on Windows)
Write-Step "Installing llama-cpp-python... (this may take a while)"
try {
  # Prefer wheels to avoid CMake/MSVC builds on Windows. Wheels exist for Python 3.11/3.12.
  uv pip install --python $venvPython --upgrade --verbose --force-reinstall --no-cache-dir --only-binary=:all: `
    llama-cpp-python==0.3.5
} catch {
  Write-Err "Failed to install llama-cpp-python prebuilt wheel."
  Write-Info "If you need local llama.cpp, install 'Visual Studio 2022 Build Tools' with the C++ workload, then re-run. Continuing without llama-cpp-python."
}

# 7) Start the server
Write-Host "`nStarting Morphik server...`n" -ForegroundColor Green
uv run --python $venvPython start_server.py


