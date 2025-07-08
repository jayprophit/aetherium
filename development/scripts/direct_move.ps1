# Direct Move Script - Uses .NET methods to move directories
# Run with: .\scripts\direct_move.ps1

Add-Type -AssemblyName System.IO

$rootDir = Split-Path -Parent $PSScriptRoot
$logFile = Join-Path $rootDir "direct_move_log_$(Get-Date -Format 'yyyyMMdd_HHmmss').txt"

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

Write-Log "Starting direct directory moves..."

# Process each move
foreach ($move in $moves) {
    $source = [System.IO.Path]::Combine($rootDir, $move.From)
    $destination = [System.IO.Path]::Combine($rootDir, $move.To)
    $destinationParent = [System.IO.Path]::GetDirectoryName($destination)
    
    Write-Log "Processing: $($move.From) -> $($move.To)"
    
    # Skip if source doesn't exist
    if (-not [System.IO.Directory]::Exists($source)) {
        Write-Log "  WARNING: Source not found"
        continue
    }
    
    try {
        # Create parent directory if it doesn't exist
        if (-not [System.IO.Directory]::Exists($destinationParent)) {
            [System.IO.Directory]::CreateDirectory($destinationParent) | Out-Null
            Write-Log "  Created directory: $destinationParent"
        }
        
        # If destination exists, merge contents
        if ([System.IO.Directory]::Exists($destination)) {
            Write-Log "  Merging with existing directory at destination"
            
            # Get all items in source
            $items = [System.IO.Directory]::GetFileSystemEntries($source)
            $itemCount = $items.Count
            $movedCount = 0
            $skippedCount = 0
            $errorCount = 0
            
            foreach ($item in $items) {
                $itemName = [System.IO.Path]::GetFileName($item)
                $itemDest = [System.IO.Path]::Combine($destination, $itemName)
                
                if (-not [System.IO.Directory]::Exists($itemDest) -and -not [System.IO.File]::Exists($itemDest)) {
                    try {
                        if ([System.IO.Directory]::Exists($item)) {
                            [System.IO.Directory]::Move($item, $itemDest)
                        } else {
                            [System.IO.File]::Move($item, $itemDest)
                        }
                        $movedCount++
                    } catch {
                        $errorCount++
                        Write-Log "    ERROR: Failed to move $itemName - $_"
                    }
                } else {
                    $skippedCount++
                    Write-Log "    WARNING: Skipping existing item: $itemName"
                }
            }
            
            # Try to remove source directory if empty
            try {
                if ($movedCount -gt 0) {
                    $remainingItems = [System.IO.Directory]::GetFileSystemEntries($source)
                    if ($remainingItems.Count -eq 0) {
                        [System.IO.Directory]::Delete($source, $false)
                        Write-Log "  Removed empty source directory: $($move.From)"
                    } else {
                        Write-Log "  WARNING: Could not remove source directory (not empty): $($move.From)"
                    }
                }
            } catch {
                Write-Log "  WARNING: Could not remove source directory: $($move.From) - $_"
            }
            
            Write-Log "  Moved $movedCount items, skipped $skippedCount items, $errorCount errors"
            
        } else {
            # Move the entire directory
            [System.IO.Directory]::Move($source, $destination)
            Write-Log "  SUCCESS: Moved to $($move.To)"
        }
        
    } catch {
        Write-Log "  ERROR: Failed to process - $_"
        
        # Try one more time with a different method
        try {
            Write-Log "  Trying alternative method..."
            $tempDest = "${destination}_temp_$(Get-Random)"
            
            # Use robocopy to move files
            $robocopyArgs = @("$source", "$tempDest", "/E", "/MOVE", "/NFL", "/NDL", "/NJH", "/NJS", "/NC", "/NS", "/NP", "/R:1", "/W:1")
            Start-Process -FilePath "robocopy.exe" -ArgumentList $robocopyArgs -Wait -NoNewWindow
            
            if ($LASTEXITCODE -lt 8) {
                # Success - rename temp to final destination
                [System.IO.Directory]::Move($tempDest, $destination)
                Write-Log "  SUCCESS: Moved using robocopy to $($move.To)"
            } else {
                # Clean up temp directory on failure
                if ([System.IO.Directory]::Exists($tempDest)) {
                    [System.IO.Directory]::Delete($tempDest, $true)
                }
                Write-Log "  ERROR: Robocopy failed with exit code $LASTEXITCODE"
            }
        } catch {
            Write-Log "  ERROR: Alternative method also failed - $_"
        }
    }
}

# Final report of remaining top-level directories
$protectedDirs = @('.git', '.venv', '.devcontainer', 'docs', 'development', 'operations', 'resources', 'src', 'systems', 'examples')
$allDirs = [System.IO.Directory]::GetDirectories($rootDir)
$topLevelDirs = $allDirs | Where-Object { 
    $dirName = [System.IO.Path]::GetFileName($_)
    $protectedDirs -notcontains $dirName
}

if ($topLevelDirs) {
    Write-Log "`nThe following top-level directories remain and may need attention:"
    foreach ($dir in $topLevelDirs) {
        $dirName = [System.IO.Path]::GetFileName($dir)
        $size = [math]::Round(([System.IO.Directory]::GetFiles($dir, "*.*", [System.IO.SearchOption]::AllDirectories) | 
            Measure-Object -Property Length -Sum).Sum / 1KB, 2)
        $fileCount = [System.IO.Directory]::GetFiles($dir, "*.*", [System.IO.SearchOption]::AllDirectories).Count
        Write-Log "- $dirName (Size: ${size}KB, Files: $fileCount)"
    }
}

# List remaining files in root
$allFiles = [System.IO.Directory]::GetFiles($rootDir)
$protectedFiles = @('.env', 'Dockerfile', 'LICENSE', 'README.md', 'package.json', 'index.html', 'validation_output.txt', 'validation_results.txt')
$rootFiles = $allFiles | Where-Object { 
    $fileName = [System.IO.Path]::GetFileName($_)
    $protectedFiles -notcontains $fileName
}

if ($rootFiles) {
    Write-Log "`nThe following files remain in the root directory and may need attention:"
    foreach ($file in $rootFiles) {
        $fileInfo = New-Object System.IO.FileInfo($file)
        Write-Log "- $($fileInfo.Name) (Size: $([math]::Round($fileInfo.Length / 1KB, 2))KB)"
    }
}

Write-Log "`nDirect move operation complete! See $logFile for details."
