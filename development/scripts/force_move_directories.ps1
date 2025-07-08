# Force move directories script - handles stubborn directories
# Run with: .\scripts\force_move_directories.ps1

$rootDir = Split-Path -Parent $PSScriptRoot
$logFile = Join-Path $rootDir "force_move_log_$(Get-Date -Format 'yyyyMMdd_HHmmss').txt"

# Function to log messages
function Write-Log {
    param([string]$message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] $message"
    Add-Content -Path $logFile -Value $logMessage
    Write-Host $logMessage
}

# Define directory moves
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

Write-Log "Starting forced directory moves..."

# Process each move
foreach ($move in $moves) {
    $source = Join-Path $rootDir $move.From
    $destination = Join-Path $rootDir $move.To
    $destinationParent = Split-Path $destination -Parent
    
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
    
    # If destination exists, merge contents
    if (Test-Path $destination) {
        try {
            Write-Log "  Merging with existing directory at destination"
            
            # Get all items in source
            $items = Get-ChildItem -Path $source -Force
            $itemCount = $items.Count
            $movedCount = 0
            $skippedCount = 0
            $errorCount = 0
            
            foreach ($item in $items) {
                $itemDest = Join-Path $destination $item.Name
                if (-not (Test-Path $itemDest)) {
                    try {
                        Move-Item -Path $item.FullName -Destination $destination -Force -ErrorAction Stop
                        $movedCount++
                    } catch {
                        $errorCount++
                        Write-Log "    ERROR: Failed to move $($item.Name) - $_"
                    }
                } else {
                    $skippedCount++
                    Write-Log "    WARNING: Skipping existing item: $($item.Name)"
                }
            }
            
            # Try to remove source directory if empty
            try {
                if ($movedCount -gt 0) {
                    $remainingItems = @(Get-ChildItem -Path $source -Force)
                    if ($remainingItems.Count -eq 0) {
                        Remove-Item -Path $source -Force -Recurse -ErrorAction Stop
                        Write-Log "  Removed empty source directory: $($move.From)"
                    } else {
                        Write-Log "  WARNING: Could not remove source directory (not empty): $($move.From)"
                    }
                }
            } catch {
                Write-Log "  WARNING: Could not remove source directory: $($move.From) - $_"
            }
            
            Write-Log "  Moved $movedCount items, skipped $skippedCount items, $errorCount errors"
            
        } catch {
            Write-Log "  ERROR: Failed to merge directories - $_"
        }
    } else {
        # Move the entire directory
        try {
            Move-Item -Path $source -Destination $destination -Force -ErrorAction Stop
            Write-Log "  SUCCESS: Moved to $($move.To)"
        } catch {
            Write-Log "  ERROR: Failed to move - $_"
            
            # Try alternative method using robocopy
            try {
                Write-Log "  Trying alternative method with robocopy..."
                $tempDest = "${destination}_temp_$(Get-Random)"
                
                # Copy all files
                & robocopy $source $tempDest /E /MOVE /NFL /NDL /NJH /NJS /NC /NS /NP /R:1 /W:1
                
                if ($LASTEXITCODE -lt 8) {
                    # Success - rename temp to final destination
                    Rename-Item -Path $tempDest -NewName (Split-Path $destination -Leaf) -Force
                    Write-Log "  SUCCESS: Moved using robocopy to $($move.To)"
                } else {
                    # Clean up temp directory on failure
                    if (Test-Path $tempDest) {
                        Remove-Item -Path $tempDest -Recurse -Force -ErrorAction SilentlyContinue
                    }
                    Write-Log "  ERROR: Robocopy failed with exit code $LASTEXITCODE"
                }
            } catch {
                Write-Log "  ERROR: Robocopy also failed - $_"
            }
        }
    }
}

# Final report of remaining top-level directories
$protectedDirs = @('.git', '.venv', '.devcontainer', 'docs', 'development', 'operations', 'resources', 'src', 'systems', 'examples')
$topLevelDirs = Get-ChildItem -Path $rootDir -Directory | 
    Where-Object { $_.Name -notin $protectedDirs }

if ($topLevelDirs) {
    Write-Log "`nThe following top-level directories remain and may need attention:"
    $topLevelDirs | ForEach-Object {
        $size = if ($_.PSIsContainer) { 
            "$([math]::Round((Get-ChildItem $_.FullName -Recurse -File | Measure-Object -Property Length -Sum).Sum / 1KB, 2)) KB" 
        } else { "N/A" }
        $itemCount = (Get-ChildItem $_.FullName -Recurse -File).Count
        Write-Log "- $($_.Name) (Type: $($_.GetType().Name), Size: $size, Files: $itemCount)"
    }
}

# List remaining files in root
$rootFiles = Get-ChildItem -Path $rootDir -File | 
    Where-Object { $_.Name -notin @('.env', 'Dockerfile', 'LICENSE', 'README.md', 'package.json', 'index.html', 'validation_output.txt', 'validation_results.txt') }

if ($rootFiles) {
    Write-Log "`nThe following files remain in the root directory and may need attention:"
    $rootFiles | ForEach-Object {
        Write-Log "- $($_.Name) (Size: $([math]::Round($_.Length / 1KB, 2)) KB)"
    }
}

Write-Log "`nForce move operation complete! See $logFile for details."
