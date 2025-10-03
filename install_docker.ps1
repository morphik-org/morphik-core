<#
  Morphik Core - Docker installer for Windows PowerShell.

  This mirrors install_docker.sh (macOS/Linux) and sets up a Docker-based deployment.

  Usage (PowerShell):
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
    ./install_docker.ps1

  Requirements:
    - Docker Desktop running (Compose V2 included)
    - Internet connectivity to pull the image or fetch config files
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$script:EmbeddingSelection = $null

function Write-Info($msg)  { Write-Host "[INFO]  $msg" -ForegroundColor Cyan }
function Write-Step($msg)  { Write-Host "[STEP]  $msg" -ForegroundColor Yellow }
function Write-Ok($msg)    { Write-Host "[OK]    $msg" -ForegroundColor Green }
function Write-Err($msg)   { Write-Host "[ERROR] $msg" -ForegroundColor Red }

$REPO_URL   = "https://raw.githubusercontent.com/morphik-org/morphik-core/main"
$COMPOSE    = "docker-compose.run.yml"
$IMAGE      = "ghcr.io/morphik-org/morphik-core:latest"
$DIRECT_URL = "https://www.morphik.ai/docs/getting-started#self-host-direct-installation-advanced"

function Assert-Docker {
  if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Err "Docker is required. Please install Docker Desktop and re-run."
    throw "Docker not found"
  }
  cmd.exe /c "docker info >NUL 2>&1"
  if ($LASTEXITCODE -ne 0) {
    Write-Err "Docker is installed but not running. Start Docker Desktop and retry."
    throw "Docker not running"
  }
  try { docker compose version | Out-Null } catch {
    Write-Err "Docker Compose V2 is required. Please update Docker Desktop."
    throw "Compose V2 missing"
  }
}

function New-RandomHex($bytes) {
  $buffer = New-Object byte[] $bytes
  $rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
  try {
    $rng.GetBytes($buffer)
  } finally {
    $rng.Dispose()
  }
  ($buffer | ForEach-Object { $_.ToString('x2') }) -join ''
}

function Download-File($url, $outPath) {
  Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $outPath -ErrorAction Stop
}

function Ensure-ComposeFile {
  Write-Step "Downloading the Docker Compose configuration file..."
  try {
    Download-File "$REPO_URL/$COMPOSE" $COMPOSE
    Write-Ok "Downloaded '$COMPOSE'."
  } catch {
    Write-Err "Failed to download '$COMPOSE'. Check connectivity and try again."
    throw
  }
}

function Ensure-EnvFile {
  Write-Step "Creating '.env' file for secrets..."
  $jwt = "your-super-secret-key-$(New-RandomHex 16)"
  $envContent = @(
    "# Your OpenAI API key (optional - you can configure other providers in morphik.toml)",
    "OPENAI_API_KEY=",
    "",
    "# A secret key for signing JWTs. A random one is generated for you.",
    "JWT_SECRET_KEY=$jwt",
    "",
    "# Local URI token for secure URI generation (required for creating connection URIs)",
    "LOCAL_URI_TOKEN="
  ) -join [Environment]::NewLine
  Set-Content -Path .env -Value $envContent

  $openai = Read-Host "Enter your OpenAI API Key (or press Enter to skip)"
  if ($openai) {
    (Get-Content .env -Raw) -replace "OPENAI_API_KEY=", "OPENAI_API_KEY=$openai" |
      Set-Content .env
    Write-Ok "Configured OPENAI_API_KEY in .env"
  } else {
    Write-Info "No OpenAI API key provided. You can configure providers in morphik.toml later."
    Write-Info "Embeddings power ingestion, search, and querying in Morphik. Choose an alternative provider to continue."
    while ($true) {
      Write-Host ""
      Write-Host "  1) Lemonade embeddings (requires Lemonade SDK running locally)"
      Write-Host "  2) Ollama embeddings (default - requires Ollama with nomic-embed-text installed)"
      $choice = Read-Host "Enter 1 or 2 [2]"
      if ([string]::IsNullOrWhiteSpace($choice)) { $choice = '2' }
      switch ($choice) {
        '1' {
          $script:EmbeddingSelection = [pscustomobject]@{ Model = 'lemonade_embedding'; Label = 'Lemonade embeddings' }
          break
        }
        '2' {
          $script:EmbeddingSelection = [pscustomobject]@{ Model = 'ollama_embedding'; Label = 'Ollama embeddings' }
          break
        }
        default {
          Write-Step "Please enter 1 or 2."
        }
      }
    }
    Write-Info ("Embeddings will be configured to use {0} with 768 dimensions." -f $script:EmbeddingSelection.Label)
    if ($script:EmbeddingSelection.Model -eq 'lemonade_embedding') {
      Write-Step "Ensure the Lemonade SDK is installed and running (you'll see an installer prompt later in this script)."
    }
  }
}

function Update-EmbeddingConfig {
  param(
    [Parameter(Mandatory)] [string] $Model,
    [Parameter(Mandatory)] [string] $Label
  )

  if (-not (Test-Path 'morphik.toml')) {
    Write-Step "morphik.toml not found. Skipping embedding configuration update."
    return
  }

  Write-Step ("Configuring morphik.toml to use {0} (768 dimensions)..." -f $Label)
  $lines = Get-Content 'morphik.toml'
  $inEmbedding = $false
  for ($i = 0; $i -lt $lines.Length; $i++) {
    $line = $lines[$i]
    if ($line -match '^\s*\[embedding\]\s*$') { $inEmbedding = $true; continue }
    if ($inEmbedding -and $line -match '^\s*\[') { $inEmbedding = $false }
    if (-not $inEmbedding) { continue }

    if ($line -match '^\s*model\s*=\s*"([^"]*)"(.*)') {
      $comment = $Matches[2]
      $lines[$i] = 'model = "{0}"{1}' -f $Model, $comment
    } elseif ($line -match '^\s*dimensions\s*=\s*\d+(.*)') {
      $suffix = $Matches[1]
      $lines[$i] = "dimensions = 768$suffix"
    }
  }

  Set-Content -Path 'morphik.toml' -Value $lines
}

function Try-Extract-Config {
  Write-Step "Pulling Docker image (this may take several minutes)..."

  $pulled = $true
  try {
    # Use cmd to ensure proper progress display in PowerShell
    cmd /c "docker pull $IMAGE"
    if ($LASTEXITCODE -eq 0) {
      $pulled = $true
      Write-Ok "Docker image pulled successfully."
    } else {
      $pulled = $false
      Write-Err "Docker pull failed with exit code $LASTEXITCODE"
    }
  } catch {
    Write-Err "Failed to pull Docker image: $($_.Exception.Message)"
    $pulled = $false
  }

  if ($pulled) {
    Write-Ok "Docker image is available. Extracting default 'morphik.toml'..."
    $content = ''
    try { $content = docker run --rm $IMAGE cat /app/morphik.toml.default }
    catch { $content = '' }

    if ($content) {
      Set-Content -Path morphik.toml -Value $content
      if ((Test-Path morphik.toml) -and ((Get-Item morphik.toml).Length -gt 0)) {
        Write-Ok "Extracted configuration from Docker image."
        return
      }
    }

    Write-Info "Trying alternative extraction method (docker cp)..."
    $cid = ''
    try { $cid = docker create $IMAGE } catch { $cid = '' }
    if ($cid) {
      try {
        docker cp "$cid`:/app/morphik.toml.default" morphik.toml | Out-Null
      } catch { }
      try { docker rm $cid | Out-Null } catch { }
      if ((Test-Path morphik.toml) -and ((Get-Item morphik.toml).Length -gt 0)) {
        Write-Ok "Extracted configuration using docker cp."
        return
      }
    }
  } else {
    Write-Info "Failed to pull image. Will download configuration from repository instead."
  }

  Write-Step "Downloading configuration from repository..."
  $downloaded = $false
  try {
    Download-File "$REPO_URL/morphik.docker.toml" "morphik.toml"
    $downloaded = $true
    Write-Ok "Downloaded Docker-specific configuration."
  } catch {
    try {
      Download-File "$REPO_URL/morphik.toml" "morphik.toml"
      $downloaded = $true
      Write-Info "Downloaded standard morphik.toml (may need Docker adjustments)."
    } catch {
      Write-Err "Could not obtain a configuration file."
      throw
    }
  }
}

function Update-DevMode-Or-Auth {
  Write-Host ""; Write-Info "Setting up authentication for your Morphik deployment:"
  Write-Info " • For external access, set a LOCAL_URI_TOKEN."
  Write-Info " • For local-only access, press Enter to enable dev_mode."
  $token = Read-Host "Enter a secure LOCAL_URI_TOKEN (or press Enter to skip)"
  if ([string]::IsNullOrWhiteSpace($token)) {
    Write-Info "No token provided - enabling development mode (dev_mode=true)."
    $content = Get-Content morphik.toml -Raw
    $content = $content -replace '(?m)^dev_mode\s*=\s*false', 'dev_mode = true'
    Set-Content morphik.toml -Value $content
  } else {
    Write-Ok "LOCAL_URI_TOKEN set - keeping production mode (dev_mode=false)."
    (Get-Content .env -Raw) -replace 'LOCAL_URI_TOKEN=', "LOCAL_URI_TOKEN=$token" |
      Set-Content .env
  }
}

function Update-GPU-Options {
  Write-Host ""; Write-Info "Multimodal embeddings can use GPU for best accuracy."
  $gpu = Read-Host "Do you have a GPU available for Morphik to use? (y/N)"
  if ($gpu -notin @('y','Y')) {
    Write-Info "Disabling multimodal embeddings and reranking (CPU-only)."
    $cfg = Get-Content morphik.toml -Raw
    $cfg = $cfg -replace '(?m)^enable_colpali\s*=\s*true', 'enable_colpali = false'
    $cfg = $cfg -replace '(?m)^use_reranker\s*=\s*.*', 'use_reranker = false'
    Set-Content morphik.toml -Value $cfg
    Write-Ok "Configuration updated for CPU-only operation."
  } else {
    Write-Ok "GPU selected. Multimodal embeddings will remain enabled."
  }
}

function Enable-Config-Mount {
  Write-Step "Enabling configuration mounting in '$COMPOSE'..."
  $composeContent = Get-Content $COMPOSE -Raw
  $composeContent = $composeContent -replace '#\s*-\s*\./morphik.toml:/app/morphik.toml:ro',
                                 '- ./morphik.toml:/app/morphik.toml:ro'
  [System.IO.File]::WriteAllText($COMPOSE, $composeContent, (New-Object System.Text.UTF8Encoding($false)))
}

function Get-ApiPortFromToml {
  $lines = Get-Content morphik.toml -ErrorAction Stop
  $inApi = $false
  $port  = $null
  foreach ($line in $lines) {
    if ($line -match '^\s*\[api\]\s*$') { $inApi = $true; continue }
    if ($inApi -and $line -match '^\s*\[') { break }
    if ($inApi -and $line -match '^\s*port\s*=\s*"?(\d+)"?') {
      $port = $Matches[1]; break
    }
  }
  if (-not $port) { $port = '8000' }
  return $port
}

function Update-Port-Mapping($apiPort) {
  $composeContent = Get-Content $COMPOSE -Raw
  $composeContent = $composeContent -replace '"8000:8000"', '"{0}:{0}"' -f $apiPort
  [System.IO.File]::WriteAllText($COMPOSE, $composeContent, (New-Object System.Text.UTF8Encoding($false)))
}

function Maybe-Install-UI($apiPort) {
  Write-Host ""; Write-Info "Morphik includes an optional Admin UI."
  $ans = Read-Host "Would you like to install the Admin UI? (y/N)"
  if ($ans -in @('y','Y')) {
    Write-Step "Extracting UI component files from Docker image..."
    $cid = ''
    try { $cid = docker create $IMAGE } catch { $cid = '' }
    if ($cid) {
      $ok = $true
      try {
        if (-not (Test-Path "ee")) { New-Item -ItemType Directory -Path "ee" | Out-Null }
        docker cp "$cid`:/app/ee/ui-component" "ee/ui-component" | Out-Null
      } catch { $ok = $false }
      try { docker rm $cid | Out-Null } catch { }
      if ($ok -and (Test-Path "ee/ui-component")) {
        Write-Ok "UI component downloaded successfully."
        (Get-Content .env -Raw) + [Environment]::NewLine + "UI_INSTALLED=true" |
          Set-Content .env
        $composeContent = Get-Content $COMPOSE -Raw
        $composeContent = $composeContent -replace 'NEXT_PUBLIC_API_URL=http://localhost:8000',
                                     ('NEXT_PUBLIC_API_URL=http://localhost:{0}' -f $apiPort)
        [System.IO.File]::WriteAllText($COMPOSE, $composeContent, (New-Object System.Text.UTF8Encoding($false)))
        return $true
      } else {
        Write-Err "Failed to download UI component. Continuing without UI."
      }
    } else {
      Write-Err "Could not create a container to extract UI. Continuing without UI."
    }
  }
  return $false
}

function Start-Stack($apiPort, $ui) {
  Write-Step "Starting the Morphik stack... (first run can take a few minutes)"
  $args = @('-f', $COMPOSE)
  if ($ui) { $args += @('--profile','ui') }
  docker compose @args up -d
  Write-Ok "Morphik has been started!"
  Write-Info ("Health check: http://localhost:{0}/health" -f $apiPort)
  Write-Info ("API docs:     http://localhost:{0}/docs"   -f $apiPort)
  Write-Info ("Main API:     http://localhost:{0}"        -f $apiPort)
  if ($ui) {
    Write-Info "Admin UI:     http://localhost:3003"
  }

  # Create convenience startup script for Windows
  $start = @(
    "Set-StrictMode -Version Latest",
    "`$ErrorActionPreference = 'Stop'",
    "",
    'function Write-Info($msg) { Write-Host "[INFO]  $msg" -ForegroundColor Cyan }',
    'function Write-Warn($msg) { Write-Host "[WARN]  $msg" -ForegroundColor Yellow }',
    "",
    "`$apiPortVar = `"$apiPort`"",
    "# Read desired port from morphik.toml if available",
    "`$desired = `$apiPortVar",
    "if (Test-Path 'morphik.toml') {",
    "  `$lines = Get-Content morphik.toml",
    "  `$inApi = `$false; `$p = `$null",
    "  foreach (`$l in `$lines) {",
    "    if (`$l -match '^\\s*\\[api\\]\\s*`$') { `$inApi = `$true; continue }",
    "    if (`$inApi -and `$l -match '^\\s*\\[') { break }",
    "    if (`$inApi -and `$l -match '^\\s*port\\s*=\\s*`"?(\\d+)`"?') {",
    "      `$p = `$Matches[1]; break } }",
    "  if (`$p) { `$desired = `$p }",
    "}",
    "",
    "# Update port mapping in compose file if needed",
    "`$compose = Get-Content 'docker-compose.run.yml' -Raw",
    "if (`$compose -match '`"(\\d+):(\\d+)`"') {",
    "  `$current = `$Matches[1]",
    "  if (`$current -ne `$desired) {",
    "    `$compose = `$compose -replace `"`$(`$current`):`$(`$current`)`", `"`$(`$desired`):`$(`$desired`)`"",
    "    Set-Content 'docker-compose.run.yml' -Value `$compose  } }",
    "",
    "# Warn if multimodal embeddings disabled",
    "if (Test-Path 'morphik.toml') {",
    "  `$cfg = Get-Content morphik.toml -Raw",
    "  if (`$cfg -match '(?m)^enable_colpali\\s*=\\s*false') {",
    "    Write-Warn 'Multimodal embeddings are disabled. Enable in morphik.toml if you have a GPU.' } }",
    "",
    "# Include UI profile if installed",
    "`$ui = `$false",
    "if (Test-Path '.env') {",
    "  `$envText = Get-Content .env -Raw",
    "  if (`$envText -match 'UI_INSTALLED=true') { `$ui = `$true } }",
    "",
    "`$args = @('-f','docker-compose.run.yml')",
    "if (`$ui) { `$args += @('--profile','ui') }",
    "docker compose @args up -d",
    "Write-Host `"Morphik is running on http://localhost:`$(`$desired)`""
  ) -join [Environment]::NewLine
  Set-Content -Path 'start-morphik.ps1' -Value $start }

# --- Main ---
Write-Info "Checking for Docker and Docker Compose..."
Assert-Docker
Write-Ok "Prerequisites are satisfied."

# Apple Silicon note (informational)
if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64') {
  Write-Host ""
  Write-Info "You appear to be on ARM64. For best performance with GPU, consider Direct Installation:"
  Write-Info $DIRECT_URL
}

Ensure-ComposeFile
Ensure-EnvFile
Try-Extract-Config
if ($script:EmbeddingSelection) {
  Update-EmbeddingConfig -Model $script:EmbeddingSelection.Model -Label $script:EmbeddingSelection.Label
}
Update-DevMode-Or-Auth
Update-GPU-Options
Enable-Config-Mount

$apiPort = Get-ApiPortFromToml
Update-Port-Mapping -apiPort $apiPort

$uiInstalled = Maybe-Install-UI -apiPort $apiPort
Start-Stack -apiPort $apiPort -ui $uiInstalled

Write-Host ""
Write-Ok "Management commands:"
Write-Info "View logs:    docker compose -f $COMPOSE $(if($uiInstalled){'--profile ui '})logs -f"
Write-Info "Stop services: docker compose -f $COMPOSE $(if($uiInstalled){'--profile ui '})down"
Write-Info "Restart:      ./start-morphik.ps1"

Write-Host ""
Write-Ok "Enjoy using Morphik!"
