# Force reorganization script with detailed logging and individual moves
# Run with: .\scripts\force_reorganize.ps1

$rootDir = Split-Path -Parent $PSScriptRoot
$logFile = Join-Path $rootDir "force_reorganize_$(Get-Date -Format 'yyyyMMdd_HHmmss').txt"

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

Write-Log "Starting forced reorganization..."

# Process each move individually
foreach ($move in $moves) {
    $source = Join-Path $rootDir $move.From
    $destination = Join-Path $rootDir $move.To
    $destinationParent = Split-Path $destination -Parent
    $newName = Split-Path $destination -Leaf
    
    Write-Log "Processing: $($move.From) -> $($move.To)"
    
    # Skip if source doesn't exist
    if (-not (Test-Path $source)) {
        Write-Log "  WARNING: Source not found"
        continue
    }
    
    # Create parent directory if it doesn't exist
    if (-not (Test-Path $destinationParent)) {
        try {
            New-Item -ItemType Directory -Path $destinationParent -Force | Out-Null
            Write-Log "  Created directory: $destinationParent"
        } catch {
            Write-Log "  ERROR: Failed to create directory $destinationParent - $_"
            continue
        }
    }
    
    # Handle existing destination
    if (Test-Path $destination) {
        try {
            # If it's a directory, merge contents
            if ((Get-Item $source).PSIsContainer -and (Test-Path $destination -PathType Container)) {
                Write-Log "  Merging with existing directory at destination"
                Get-ChildItem -Path $source | ForEach-Object {
                    $itemDest = Join-Path $destination $_.Name
                    if (Test-Path $itemDest) {
                        Write-Log "    WARNING: Skipping existing item: $($_.Name)"
                    } else {
                        Move-Item -Path $_.FullName -Destination $destination -Force
                        Write-Log "    Moved: $($_.Name)"
                    }
                }
                # Remove source if empty
                if (-not (Get-ChildItem -Path $source -Force)) {
                    Remove-Item -Path $source -Force
                }
            } else {
                Write-Log "  WARNING: Destination already exists and is not a directory, skipping"
            }
            continue
        } catch {
            Write-Log "  ERROR: Failed to handle existing destination - $_"
            continue
        }
    }
    
    # Perform the move
    try {
        Move-Item -Path $source -Destination $destination -Force
        Write-Log "  SUCCESS: Moved to $($move.To)"
    } catch {
        Write-Log "  ERROR: Failed to move - $_"
    }
}

# Final check for remaining top-level directories
$protectedDirs = @('.git', '.venv', '.devcontainer', 'docs', 'development', 'operations', 'resources', 'src', 'systems', 'examples')
$topLevelDirs = Get-ChildItem -Path $rootDir -Directory | 
    Where-Object { $_.Name -notin $protectedDirs }

if ($topLevelDirs) {
    Write-Log "`nThe following top-level directories remain and may need attention:"
    $topLevelDirs | ForEach-Object {
        $size = if ($_.PSIsContainer) { 
            "$((Get-ChildItem $_.FullName -Recurse -File | Measure-Object -Property Length -Sum).Sum / 1KB) KB" 
        } else { "N/A" }
        Write-Log "- $($_.Name) (Type: $($_.GetType().Name), Size: $size)"
    }
}

Write-Log "`nReorganization complete! See $logFile for details."
