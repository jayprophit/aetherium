# Enhanced directory reorganization script with better error handling and logging
# Run with: .\scripts\enhanced_reorganize.ps1

$rootDir = Split-Path -Parent $PSScriptRoot
$logFile = Join-Path $rootDir "reorganization_log_$(Get-Date -Format 'yyyyMMdd_HHmmss').txt"

function Write-Log {
    param([string]$message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] $message"
    Add-Content -Path $logFile -Value $logMessage
    Write-Host $logMessage
}

# Define all moves with source and destination
$moves = @(
    # Resources
    @{ From = "ai"; To = "resources\ai" },
    @{ From = "applications"; To = "resources\applications" },
    @{ From = "cross-platform"; To = "resources\cross-platform" },
    @{ From = "domains"; To = "resources\domains" },
    @{ From = "movement"; To = "resources\movement" },
    @{ From = "networking"; To = "resources\networking" },
    @{ From = "perception"; To = "resources\perception" },
    @{ From = "smart-devices"; To = "resources\smart-devices" },
    @{ From = "template resources"; To = "resources\templates" },
    
    # Development
    @{ From = "scripts"; To = "development\scripts" },
    @{ From = "tests"; To = "development\tests" },
    @{ From = "maintenance"; To = "development\maintenance" },
    @{ From = "performance"; To = "development\performance" },
    
    # Documentation
    @{ From = "guides"; To = "docs\guides" },
    @{ From = "contributing"; To = "docs\contributing" },
    @{ From = "governance"; To = "docs\governance" },
    @{ From = "meta"; To = "docs\meta" },
    
    # Systems
    @{ From = "platform"; To = "systems\platform" },
    @{ From = "platforms"; To = "systems\platforms" },
    @{ From = "protocols"; To = "systems\protocols" },
    @{ From = "core"; To = "systems\core" },
    @{ From = "mcp"; To = "systems\protocols\mcp" },
    
    # Operations
    @{ From = "iac"; To = "operations\iac" },
    @{ From = "process"; To = "operations\process" },
    
    # Special cases
    @{ From = "keras_project"; To = "examples\keras" },
    @{ From = "native"; To = "src\native" },
    @{ From = "api"; To = "src\api" }
)

# Create target directories if they don't exist
Write-Log "Creating target directories..."
$moves | ForEach-Object {
    $targetDir = Join-Path $rootDir (Split-Path $_.To -Parent)
    if (-not (Test-Path $targetDir)) {
        try {
            New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
            Write-Log "Created directory: $targetDir"
        } catch {
            Write-Log "ERROR: Failed to create directory $targetDir - $_"
        }
    }
}

# Process each move
Write-Log "Starting directory moves..."
$moves | ForEach-Object {
    $source = Join-Path $rootDir $_.From
    $destination = Join-Path $rootDir $_.To
    $destinationParent = Split-Path $destination -Parent
    $newName = Split-Path $destination -Leaf
    
    if (-not (Test-Path $source)) {
        Write-Log "WARNING: Source not found: $($_.From)"
        return
    }
    
    if (Test-Path $destination) {
        Write-Log "WARNING: Destination already exists, skipping: $($_.To)"
        return
    }
    
    try {
        # Move the item to the parent directory first
        $tempPath = Join-Path $destinationParent (Split-Path $source -Leaf)
        Move-Item -Path $source -Destination $destinationParent -Force -ErrorAction Stop
        
        # Rename if needed
        if ((Split-Path $source -Leaf) -ne $newName) {
            Rename-Item -Path $tempPath -NewName $newName -ErrorAction Stop
        }
        
        Write-Log "SUCCESS: Moved $($_.From) to $($_.To)"
    } catch {
        Write-Log "ERROR: Failed to move $($_.From) to $($_.To) - $_"
    }
}

# Check for any remaining top-level directories that might need attention
$topLevelDirs = Get-ChildItem -Path $rootDir -Directory | 
    Where-Object { $_.Name -notin @('.git', '.venv', '.devcontainer', 'docs', 'development', 'operations', 'resources', 'src', 'systems') }

if ($topLevelDirs) {
    Write-Log "`nThe following top-level directories remain and may need attention:"
    $topLevelDirs | ForEach-Object {
        Write-Log "- $($_.Name) ($($_.FullName))"
    }
}

Write-Log "`nReorganization complete! See $logFile for details."
