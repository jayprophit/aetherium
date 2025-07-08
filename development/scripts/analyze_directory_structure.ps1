# Analyze Directory Structure Script
# This script analyzes the current directory structure and provides recommendations

$rootDir = Split-Path -Parent $PSScriptRoot
$reportFile = Join-Path $rootDir "directory_analysis_$(Get-Date -Format 'yyyyMMdd_HHmmss').txt"

# Function to get directory size
function Get-DirectorySize {
    param([string]$path)
    if (-not (Test-Path $path)) { return "N/A" }
    try {
        $size = (Get-ChildItem -Path $path -Recurse -File -Force -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
        if ($size -eq $null) { return "0 KB" }
        if ($size -gt 1GB) { return "{0:N2} GB" -f ($size / 1GB) }
        if ($size -gt 1MB) { return "{0:N2} MB" -f ($size / 1MB) }
        return "{0:N2} KB" -f ($size / 1KB)
    } catch {
        return "Error: $_"
    }
}

# Function to count files in directory
function Count-Files {
    param([string]$path)
    if (-not (Test-Path $path)) { return 0 }
    try {
        return (Get-ChildItem -Path $path -Recurse -File -Force -ErrorAction SilentlyContinue).Count
    } catch {
        return -1
    }
}

# Define directory mappings
$directoryMappings = @{
    # Resources
    "ai" = "resources\ai"
    "applications" = "resources\applications"
    "cross-platform" = "resources\cross-platform"
    "domains" = "resources\domains"
    "movement" = "resources\movement"
    "networking" = "resources\networking"
    "perception" = "resources\perception"
    "smart-devices" = "resources\smart-devices"
    "template resources" = "resources\templates"
    
    # Development
    "scripts" = "development\scripts"
    "tests" = "development\tests"
    "maintenance" = "development\maintenance"
    "performance" = "development\performance"
    
    # Documentation
    "guides" = "docs\guides"
    "contributing" = "docs\contributing"
    "governance" = "docs\governance"
    "meta" = "docs\meta"
    
    # Systems
    "platform" = "systems\platform"
    "platforms" = "systems\platforms"
    "protocols" = "systems\protocols"
    "core" = "systems\core"
    "mcp" = "systems\protocols\mcp"
    
    # Operations
    "iac" = "operations\iac"
    "process" = "operations\process"
    
    # Special cases
    "keras_project" = "examples\keras"
    "native" = "src\native"
    "api" = "src\api"
}

# Protected directories that should stay in root
$protectedDirs = @('.git', '.venv', '.devcontainer', 'docs', 'development', 'operations', 'resources', 'src', 'systems', 'examples')

# Protected files that should stay in root
$protectedFiles = @('.env', 'Dockerfile', 'LICENSE', 'README.md', 'package.json', 'index.html', 'validation_output.txt', 'validation_results.txt')

# Get all directories in root
$rootDirs = Get-ChildItem -Path $rootDir -Directory | Where-Object { $_.Name -notin $protectedDirs }
$rootFiles = Get-ChildItem -Path $rootDir -File | Where-Object { $_.Name -notin $protectedFiles }

# Generate report
$report = @"
# Directory Structure Analysis Report
Generated: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")

## Directories to be Moved
"@

$movesNeeded = $false
$dirTable = @()

foreach ($dir in $rootDirs) {
    $dirName = $dir.Name
    $targetPath = $directoryMappings[$dirName]
    
    if ($targetPath) {
        $fullTargetPath = Join-Path $rootDir $targetPath
        $dirSize = Get-DirectorySize -path $dir.FullName
        $fileCount = Count-Files -path $dir.FullName
        
        $dirTable += [PSCustomObject]@{
            'Directory' = $dirName
            'Current Location' = 'root'
            'Target Location' = $targetPath
            'Size' = $dirSize
            'Files' = $fileCount
            'Status' = if (Test-Path $fullTargetPath) { 'Target Exists' } else { 'Needs Moving' }
        }
        
        $movesNeeded = $true
    }
}

# Add table to report
if ($movesNeeded) {
    $report += "`n" + ($dirTable | Format-Table -AutoSize | Out-String)
} else {
    $report += "`nNo directories need to be moved.`n"
}

# Add section for remaining files
if ($rootFiles) {
    $report += "`n## Files in Root Directory (Review Needed)`n"
    $fileTable = $rootFiles | ForEach-Object {
        [PSCustomObject]@{
            'File' = $_.Name
            'Size' = "{0:N2} KB" -f ($_.Length / 1KB)
            'Last Modified' = $_.LastWriteTime.ToString("yyyy-MM-dd")
        }
    }
    $report += "`n" + ($fileTable | Format-Table -AutoSize | Out-String)
}

# Add summary
$report += @"

## Summary
- Total directories to be moved: $(($dirTable | Where-Object { $_.Status -eq 'Needs Moving' }).Count)
- Directories with existing targets: $(($dirTable | Where-Object { $_.Status -eq 'Target Exists' }).Count)
- Files in root that may need review: $($rootFiles.Count)

## Recommended Actions
1. Review the directories marked 'Needs Moving' and ensure the target locations are correct
2. For directories marked 'Target Exists', decide whether to merge or replace the existing content
3. Review files in the root directory and move them to appropriate locations if needed
4. After moving directories, update any references to the old paths in documentation and code
"@

# Save report to file
$report | Out-File -FilePath $reportFile -Encoding utf8

# Display summary
Write-Host "`nAnalysis complete! Report saved to: $reportFile"
Write-Host "Summary:"
Write-Host "- Directories to be moved: $(($dirTable | Where-Object { $_.Status -eq 'Needs Moving' }).Count)"
Write-Host "- Directories with existing targets: $(($dirTable | Where-Object { $_.Status -eq 'Target Exists' }).Count)"
Write-Host "- Files in root that may need review: $($rootFiles.Count)"
Write-Host "`nPlease review the full report for detailed information.`n"
