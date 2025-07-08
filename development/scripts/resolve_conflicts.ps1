# Resolve Directory Conflicts Script
# This script will resolve the remaining directory conflicts in the root

$rootDir = Split-Path -Parent $PSScriptRoot
$logFile = Join-Path $rootDir "resolve_conflicts_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

# Function to log messages
function Write-Log {
    param([string]$message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] $message"
    Add-Content -Path $logFile -Value $logMessage
    Write-Host $logMessage
}

# Function to safely merge directories
function Merge-Directories {
    param(
        [string]$SourcePath,
        [string]$DestPath,
        [string]$Description
    )
    
    Write-Log "Merging $Description..."
    Write-Log "  Source: $SourcePath"
    Write-Log "  Destination: $DestPath"
    
    try {
        # Create destination if it doesn't exist
        if (-not (Test-Path $DestPath)) {
            New-Item -ItemType Directory -Path $DestPath -Force | Out-Null
            Write-Log "  Created destination directory"
        }
        
        # Get all items in source
        $items = Get-ChildItem -Path $SourcePath -Force -ErrorAction SilentlyContinue
        $totalItems = $items.Count
        $movedCount = 0
        $skippedCount = 0
        $errorCount = 0
        
        foreach ($item in $items) {
            $itemDest = Join-Path $DestPath $item.Name
            
            if (Test-Path $itemDest) {
                if ($item.PSIsContainer) {
                    # If both are directories, recursively merge
                    Write-Log "    Recursively merging directory: $($item.Name)"
                    Merge-Directories -SourcePath $item.FullName -DestPath $itemDest -Description "subdirectory $($item.Name)"
                } else {
                    # File conflict - create backup
                    $backupName = "$($item.Name).backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
                    $backupPath = Join-Path $DestPath $backupName
                    Write-Log "    WARNING: File conflict for $($item.Name) - creating backup as $backupName"
                    Copy-Item -Path $item.FullName -Destination $backupPath -Force
                    $skippedCount++
                }
            } else {
                # No conflict, move the item
                try {
                    Move-Item -Path $item.FullName -Destination $DestPath -Force -ErrorAction Stop
                    $movedCount++
                    Write-Log "    Moved: $($item.Name)"
                } catch {
                    $errorCount++
                    Write-Log "    ERROR: Failed to move $($item.Name) - $_"
                }
            }
        }
        
        Write-Log "  Results: $movedCount moved, $skippedCount conflicts/backups, $errorCount errors"
        return $true
        
    } catch {
        Write-Log "  ERROR: Failed to merge directories - $_"
        return $false
    }
}

# Function to remove empty directories
function Remove-EmptyDirectory {
    param([string]$Path)
    
    try {
        $items = Get-ChildItem -Path $Path -Force -Recurse -ErrorAction SilentlyContinue
        if ($items.Count -eq 0) {
            Remove-Item -Path $Path -Force -Recurse -ErrorAction Stop
            Write-Log "  Removed empty directory: $Path"
            return $true
        } else {
            Write-Log "  WARNING: Directory not empty, cannot remove: $Path"
            return $false
        }
    } catch {
        Write-Log "  ERROR: Failed to remove directory $Path - $_"
        return $false
    }
}

Write-Log "Starting conflict resolution process..."

# Resolve applications directory conflict
$appsSource = Join-Path $rootDir "applications"
$appsTarget = Join-Path $rootDir "resources\applications"

if (Test-Path $appsSource) {
    Write-Log "Resolving applications directory conflict..."
    
    # Check for web/frontend conflict specifically
    $webSourceFrontend = Join-Path $appsSource "web\frontend"
    $webTargetFrontend = Join-Path $appsTarget "web\frontend"
    
    if ((Test-Path $webSourceFrontend) -and (Test-Path $webTargetFrontend)) {
        Write-Log "Detected web/frontend directory conflict - merging contents..."
        
        # Merge the src directories specifically
        $srcSource = Join-Path $webSourceFrontend "src"
        $srcTarget = Join-Path $webTargetFrontend "src"
        
        if ((Test-Path $srcSource) -and (Test-Path $srcTarget)) {
            if (Merge-Directories -SourcePath $srcSource -DestPath $srcTarget -Description "frontend src directories") {
                Remove-EmptyDirectory -Path $srcSource
            }
        }
        
        # Move any other items from source frontend to target frontend
        if (Merge-Directories -SourcePath $webSourceFrontend -DestPath $webTargetFrontend -Description "frontend directories") {
            Remove-EmptyDirectory -Path $webSourceFrontend
        }
    }
    
    # Now handle the entire applications directory
    if (Merge-Directories -SourcePath $appsSource -DestPath $appsTarget -Description "applications directories") {
        Remove-EmptyDirectory -Path $appsSource
    }
} else {
    Write-Log "Applications directory not found in root - already moved or doesn't exist"
}

# Resolve scripts directory conflict
$scriptsSource = Join-Path $rootDir "scripts"
$scriptsTarget = Join-Path $rootDir "development\scripts"

if (Test-Path $scriptsSource) {
    Write-Log "Resolving scripts directory conflict..."
    
    if (Merge-Directories -SourcePath $scriptsSource -DestPath $scriptsTarget -Description "scripts directories") {
        Remove-EmptyDirectory -Path $scriptsSource
    }
} else {
    Write-Log "Scripts directory not found in root - already moved or doesn't exist"
}

# Final check - list remaining non-essential directories
$protectedDirs = @('.git', '.venv', '.devcontainer', 'docs', 'development', 'operations', 'resources', 'src', 'systems', 'examples')
$remainingDirs = Get-ChildItem -Path $rootDir -Directory | 
    Where-Object { $_.Name -notin $protectedDirs } |
    Select-Object -ExpandProperty Name

if ($remainingDirs) {
    Write-Log "`nWARNING: The following directories still remain in the root:"
    $remainingDirs | ForEach-Object { 
        $dirPath = Join-Path $rootDir $_
        $size = try { 
            $sizeBytes = (Get-ChildItem -Path $dirPath -Recurse -File -Force -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
            if ($sizeBytes -gt 1GB) { "{0:N2} GB" -f ($sizeBytes / 1GB) }
            elseif ($sizeBytes -gt 1MB) { "{0:N2} MB" -f ($sizeBytes / 1MB) }
            else { "{0:N2} KB" -f ($sizeBytes / 1KB) }
        } catch { "Unknown" }
        
        $fileCount = try { 
            (Get-ChildItem -Path $dirPath -Recurse -File -Force -ErrorAction SilentlyContinue).Count 
        } catch { 0 }
        
        Write-Log "- $_ (Size: $size, Files: $fileCount)"
    }
    Write-Log "`nThese directories may need manual review or have special handling requirements."
} else {
    Write-Log "`nSUCCESS: All non-essential directories have been moved from the root!"
}

# List remaining files in root (excluding protected files)
$protectedFiles = @('.env', 'Dockerfile', 'LICENSE', 'README.md', 'package.json', 'index.html', 'validation_output.txt', 'validation_results.txt')
$remainingFiles = Get-ChildItem -Path $rootDir -File | 
    Where-Object { $_.Name -notin $protectedFiles -and -not $_.Name.EndsWith('.log') } |
    Select-Object -ExpandProperty Name

if ($remainingFiles) {
    Write-Log "`nThe following files remain in the root (excluding logs):"
    $remainingFiles | ForEach-Object { Write-Log "- $_" }
} else {
    Write-Log "`nAll non-essential files are properly located."
}

Write-Log "`nConflict resolution complete! See $logFile for details."

# Display summary
Write-Host "`nConflict resolution complete!"
Write-Host "Log file: $logFile"

if ($remainingDirs -or $remainingFiles) {
    Write-Host "`nItems that may need review:"
    if ($remainingDirs) {
        Write-Host "Directories: $($remainingDirs -join ', ')"
    }
    if ($remainingFiles) {
        Write-Host "Files: $($remainingFiles -join ', ')"
    }
} else {
    Write-Host "`nRoot directory cleanup is complete!"
}
