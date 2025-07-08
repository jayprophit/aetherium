# Final cleanup script for remaining directories
# Run with: .\scripts\final_cleanup.ps1

$rootDir = Split-Path -Parent $PSScriptRoot
$logFile = Join-Path $rootDir "final_cleanup_$(Get-Date -Format 'yyyyMMdd_HHmmss').txt"

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

Write-Log "Starting final cleanup..."

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
            $items = Get-ChildItem -Path $source -Force
            $itemCount = $items.Count
            $movedCount = 0
            
            foreach ($item in $items) {
                $itemDest = Join-Path $destination $item.Name
                if (-not (Test-Path $itemDest)) {
                    Move-Item -Path $item.FullName -Destination $destination -Force -ErrorAction Stop
                    $movedCount++
                } else {
                    Write-Log "    WARNING: Skipping existing item: $($item.Name)"
                }
            }
            
            # Remove source if empty
            if (-not (Get-ChildItem -Path $source -Force)) {
                Remove-Item -Path $source -Force
                Write-Log "  Removed empty source directory: $($move.From)"
            } else {
                Write-Log "  WARNING: Could not remove source directory (not empty): $($move.From)"
            }
            
            Write-Log "  Merged $movedCount of $itemCount items from $($move.From) to $($move.To)"
            
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
        Write-Log "- $($_.Name) (Size: $($_.Length / 1KB) KB)"
    }
}

Write-Log "`nFinal cleanup complete! See $logFile for details."
