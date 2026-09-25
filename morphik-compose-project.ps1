function Resolve-MorphikComposeProject {
  if ($env:COMPOSE_PROJECT_NAME) { return }

  $configuredProject = ''
  if (Test-Path '.env') {
    $match = Get-Content .env | Select-String '^COMPOSE_PROJECT_NAME=' | Select-Object -Last 1
    if ($match) {
      $configuredProject = $match.Line.Substring('COMPOSE_PROJECT_NAME='.Length).Trim()
    }
  }

  $containerProject = ''
  try {
    $containerProject = (& docker inspect morphik-postgres --format '{{ index .Config.Labels "com.docker.compose.project" }}' 2>$null | Select-Object -First 1).Trim()
  } catch {
    $containerProject = ''
  }
  if ($containerProject -and $containerProject -ne '<no value>') {
    $env:COMPOSE_PROJECT_NAME = $containerProject
    Write-Host "[INFO] Using existing Docker Compose project '$containerProject'." -ForegroundColor Cyan
    return
  }

  $currentProject = (Split-Path -Leaf (Get-Location).Path).ToLowerInvariant()
  $currentProject = ($currentProject -replace '[^a-z0-9_-]+', '') -replace '^[^a-z0-9]+', ''

  $candidateProject = ''
  $multipleProjects = $false
  $volumes = @()
  try {
    $volumes = @(& docker volume ls -q --filter 'label=com.docker.compose.volume=postgres_data' 2>$null)
  } catch {
    $volumes = @()
  }

  foreach ($volume in $volumes) {
    if (-not $volume) { continue }

    $volumeProject = ''
    try {
      $volumeProject = (& docker volume inspect $volume --format '{{ index .Labels "com.docker.compose.project" }}' 2>$null | Select-Object -First 1).Trim()
    } catch {
      $volumeProject = ''
    }
    if (-not $volumeProject -or $volumeProject -eq '<no value>') { continue }

    if ($configuredProject -and $volumeProject -eq $configuredProject) {
      $env:COMPOSE_PROJECT_NAME = $volumeProject
      return
    }

    if ($volumeProject -eq $currentProject) {
      $env:COMPOSE_PROJECT_NAME = $volumeProject
      Write-Host "[INFO] Using existing Docker Compose project '$volumeProject'." -ForegroundColor Cyan
      return
    }

    if (-not $candidateProject) {
      $candidateProject = $volumeProject
    } elseif ($candidateProject -ne $volumeProject) {
      $multipleProjects = $true
    }
  }

  if ($multipleProjects) {
    throw 'Multiple Morphik Postgres volumes were found. Set COMPOSE_PROJECT_NAME in .env to the project you want to use; no containers or volumes were changed.'
  }

  if ($candidateProject) {
    $env:COMPOSE_PROJECT_NAME = $candidateProject
    Write-Host "[INFO] Using existing Docker Compose project '$candidateProject'." -ForegroundColor Cyan
  } elseif ($configuredProject) {
    $env:COMPOSE_PROJECT_NAME = $configuredProject
  }
}
