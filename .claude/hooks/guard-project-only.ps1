# guard-project-only.ps1
# Purpose:
#   1. Allow normal project-local Python development.
#   2. Allow editing/deleting files inside the current project.
#   3. Block obvious destructive operations outside the current project.
#   4. Allow git push to experiment branches, but block main/master and force push.

$ErrorActionPreference = "SilentlyContinue"

$raw = [Console]::In.ReadToEnd()
if ([string]::IsNullOrWhiteSpace($raw)) {
    exit 0
}

try {
    $inputJson = $raw | ConvertFrom-Json
} catch {
    exit 0
}

# Prefer Claude project dir if available; otherwise use current working directory.
$projectRoot = $env:CLAUDE_PROJECT_DIR
if ([string]::IsNullOrWhiteSpace($projectRoot)) {
    $projectRoot = (Get-Location).Path
}

try {
    $projectRootFull = [System.IO.Path]::GetFullPath($projectRoot).TrimEnd('\','/')
} catch {
    Write-Error "Blocked: cannot resolve project root."
    exit 2
}

$toolName = [string]$inputJson.tool_name

function Resolve-ProjectPath {
    param([string]$Path)

    if ([string]::IsNullOrWhiteSpace($Path)) {
        return $null
    }

    try {
        if ([System.IO.Path]::IsPathRooted($Path)) {
            return [System.IO.Path]::GetFullPath($Path)
        } else {
            return [System.IO.Path]::GetFullPath((Join-Path $projectRootFull $Path))
        }
    } catch {
        return $null
    }
}

function Assert-InProject {
    param([string]$Path)

    $full = Resolve-ProjectPath $Path
    if ($null -eq $full) {
        Write-Error "Blocked: cannot safely resolve path: $Path"
        exit 2
    }

    $fullNorm = $full.TrimEnd('\','/')
    if (-not $fullNorm.StartsWith($projectRootFull, [System.StringComparison]::OrdinalIgnoreCase)) {
        Write-Error "Blocked: path is outside project. Path=$fullNorm Project=$projectRootFull"
        exit 2
    }
}

function Is-SecretPath {
    param([string]$Path)

    if ([string]::IsNullOrWhiteSpace($Path)) {
        return $false
    }

    $p = $Path.Replace("/", "\")
    if ($p -match '(?i)(^|\\)\.env($|\.|\\)') {
        return $true
    }
    if ($p -match '(?i)(^|\\)\.ssh(\\|$)') {
        return $true
    }
    if ($p -match '(?i)(id_rsa|id_ed25519|known_hosts)') {
        return $true
    }

    return $false
}

# Protect file tool paths.
$filePath = [string]$inputJson.tool_input.file_path
if ($filePath) {
    if (Is-SecretPath $filePath) {
        Write-Error "Blocked: secret-like file path is not allowed: $filePath"
        exit 2
    }

    # For Read/Edit/Write/MultiEdit, keep access inside the current project.
    if ($toolName -match '^(Read|Edit|Write|MultiEdit)$') {
        Assert-InProject $filePath
    }
}

# Some tool inputs may use path instead of file_path.
$path = [string]$inputJson.tool_input.path
if ($path) {
    if (Is-SecretPath $path) {
        Write-Error "Blocked: secret-like path is not allowed: $path"
        exit 2
    }

    if ($toolName -match '^(Read|Edit|Write|MultiEdit)$') {
        Assert-InProject $path
    }
}

# Protect shell commands.
$cmd = [string]$inputJson.tool_input.command
if (-not $cmd) {
    exit 0
}

# Normalize command for simple checks.
$cmdNorm = $cmd.Trim()

# Block git push to main/master and dangerous pushes.
if ($cmdNorm -match '(?i)^\s*git\s+push\b') {
    if ($cmdNorm -match '(?i)(\s|^)(--force|-f|--mirror|--all)(\s|$)') {
        Write-Error "Blocked: dangerous git push option is not allowed."
        exit 2
    }

    if ($cmdNorm -match '(?i)\borigin\s+(main|master)\b') {
        Write-Error "Blocked: pushing to main/master is not allowed."
        exit 2
    }

    if ($cmdNorm -match '(?i)\b(main|master)\s*:\s*(main|master)\b') {
        Write-Error "Blocked: pushing to main/master is not allowed."
        exit 2
    }
}

# Block obvious secret file reads through shell.
if ($cmdNorm -match '(?i)(\.env|id_rsa|id_ed25519|\.ssh)') {
    Write-Error "Blocked: command appears to access secret-like files."
    exit 2
}

# Detect destructive commands.
$isDestructive = $cmdNorm -match '(?i)(^|[;&|]\s*|\s)(rm|del|erase|rmdir|rd|Remove-Item)\b'
if (-not $isDestructive) {
    exit 0
}

# Block destructive commands targeting home/system paths.
if ($cmdNorm -match '(?i)(~|%USERPROFILE%|\$HOME|\$env:USERPROFILE|C:\\Users\\|C:\\Windows\\|C:\\Program Files|C:\\ProgramData)') {
    Write-Error "Blocked: destructive command appears to target home or system path."
    exit 2
}

# Block parent traversal in destructive commands.
if ($cmdNorm -match '(\.\./|\.\.\\)') {
    Write-Error "Blocked: destructive command uses parent-directory traversal."
    exit 2
}

# Block destructive commands targeting drive roots.
if ($cmdNorm -match '(?i)\b[A-Za-z]:\\(\s|$|["''])') {
    Write-Error "Blocked: destructive command targets a drive root."
    exit 2
}

# Resolve absolute Windows paths in destructive commands and ensure they stay inside project.
$matches = [regex]::Matches($cmdNorm, '[A-Za-z]:\\[^\s"`'']+')
foreach ($m in $matches) {
    $candidate = $m.Value.Trim('"', "'")
    try {
        $full = [System.IO.Path]::GetFullPath($candidate).TrimEnd('\','/')
        if (-not $full.StartsWith($projectRootFull, [System.StringComparison]::OrdinalIgnoreCase)) {
            Write-Error "Blocked: destructive command targets outside project: $full"
            exit 2
        }
    } catch {
        Write-Error "Blocked: cannot safely resolve destructive path: $candidate"
        exit 2
    }
}

exit 0