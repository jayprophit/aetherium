# Move Remaining Directories Script
# This script will move the remaining directories to their correct locations

# Set the root directory
$rootDir = Split-Path -Parent $PSScriptRoot
$logFile = Join-Path $rootDir "move_remaining_dirs_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

# Function to log messages
function Write-Log {
    param([string]$message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] $message"
    Add-Content -Path $logFile -Value $logMessage
    Write-Host $logMessage
}

# Define the directory moves
$directoryMoves = @(
    # Applications
    @{
        Source = "applications"
        Destination = "resources\applications"
        Action = "Move"
    },
    
    # Scripts
    @{
        Source = "scripts"
        Destination = "development\scripts"
        Action = "Merge"
    }
)

# Start the process
Write-Log "Starting directory move process..."
Write-Log "Root directory: $rootDir"

foreach ($move in $directoryMoves) {
    $sourcePath = Join-Path $rootDir $move.Source
    $destPath = Join-Path $rootDir $move.Destination
    $destParent = Split-Path -Parent $destPath
    
    Write-Log "Processing: $($move.Source) -> $($move.Destination) ($($move.Action))"
    
    # Check if source exists
    if (-not (Test-Path $sourcePath)) {
        Write-Log "  WARNING: Source directory does not exist: $sourcePath"
        continue
    }
    
    # Create parent directory if it doesn't exist
    if (-not (Test-Path $destParent)) {
        try {
            New-Item -ItemType Directory -Path $destParent -Force | Out-Null
            Write-Log "  Created directory: $destParent"
        } catch {
            Write-Log "  ERROR: Failed to create directory $destParent - $_"
            continue
        }
    }
    
    # Handle the move/merge
    try {
        if ($move.Action -eq "Move") {
            # Simple move
            if (Test-Path $destPath) {
                Write-Log "  WARNING: Destination already exists: $destPath"
                # Try to merge contents
                $items = Get-ChildItem -Path $sourcePath -Force
                $itemCount = $items.Count
                $movedCount = 0
                $skippedCount = 0
                
                foreach ($item in $items) {
                    $itemDest = Join-Path $destPath $item.Name
                    if (-not (Test-Path $itemDest)) {
                        Move-Item -Path $item.FullName -Destination $destPath -Force -ErrorAction Stop
                        $movedCount++
                    } else {
                        $skippedCount++
                        Write-Log "    WARNING: Skipping existing item: $($item.Name)"
                    }
                }
                
                # Remove source if empty
                if ((Get-ChildItem -Path $sourcePath -Force).Count -eq 0) {
                    Remove-Item -Path $sourcePath -Force -Recurse -ErrorAction SilentlyContinue
                    Write-Log "  Removed empty source directory: $sourcePath"
                } else {
                    Write-Log "  WARNING: Could not remove source directory (not empty): $sourcePath"
                }
                
                Write-Log "  Merged $movedCount items, skipped $skippedCount items"
            } else {
                # Simple move
                Move-Item -Path $sourcePath -Destination $destPath -Force -ErrorAction Stop
                Write-Log "  SUCCESS: Moved to $($move.Destination)"
            }
        }
        elseif ($move.Action -eq "Merge") {
            # Merge contents
            $items = Get-ChildItem -Path $sourcePath -Force
            $itemCount = $items.Count
            $movedCount = 0
            $skippedCount = 0
            $errorCount = 0
            
            # Ensure destination exists
            if (-not (Test-Path $destPath)) {
                New-Item -ItemType Directory -Path $destPath -Force | Out-Null
            }
            
            foreach ($item in $items) {
                $itemDest = Join-Path $destPath $item.Name
                if (-not (Test-Path $itemDest)) {
                    try {
                        Move-Item -Path $item.FullName -Destination $destPath -Force -ErrorAction Stop
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
            
            # Try to remove source if empty
            try {
                if ((Get-ChildItem -Path $sourcePath -Force -ErrorAction SilentlyContinue).Count -eq 0) {
                    Remove-Item -Path $sourcePath -Force -Recurse -ErrorAction Stop
                    Write-Log "  Removed empty source directory: $sourcePath"
                } else {
                    Write-Log "  WARNING: Could not remove source directory (not empty): $sourcePath"
                }
            } catch {
                Write-Log "  WARNING: Could not remove source directory: $sourcePath - $_"
            }
            
            Write-Log "  Merged $movedCount items, skipped $skippedCount items, $errorCount errors"
        }
    } catch {
        Write-Log "  ERROR: Failed to process - $_"
    }
}

# Final check of root directory
$protectedDirs = @('.git', '.venv', '.devcontainer', 'docs', 'development', 'operations', 'resources', 'src', 'systems', 'examples')
$remainingDirs = Get-ChildItem -Path $rootDir -Directory | 
    Where-Object { $_.Name -notin $protectedDirs } |
    Select-Object -ExpandProperty Name

if ($remainingDirs) {
    Write-Log "\nThe following directories remain in the root and may need attention:"
    $remainingDirs | ForEach-Object { Write-Log "- $_" }
} else {
    Write-Log "\nAll directories have been successfully moved!"
}

Write-Log "\nMove process complete. See $logFile for details."

# Display summary
Write-Host "\nMove process complete!"
Write-Host "Log file: $logFile"
if ($remainingDirs) {
    Write-Host "\nThe following directories remain in the root and may need attention:"
    $remainingDirs | ForEach-Object { Write-Host "- $_" }
} else {
    Write-Host "\nAll directories have been successfully moved!"
}
