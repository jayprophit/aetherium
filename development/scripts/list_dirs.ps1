# Simple script to list directories and their sizes
$rootDir = Split-Path -Parent $PSScriptRoot
$outputFile = Join-Path $rootDir "directory_listing.txt"

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

# Get all directories in root
$directories = Get-ChildItem -Path $rootDir -Directory | Sort-Object Name

# Create output
$output = "Directory Listing - $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
$output += "\n" + ("-" * 80)
$output += "\n{0,-30} {1,15} {2,10} {3}" -f "Directory", "Size", "Files", "Last Modified"
$output += "\n" + ("-" * 80)

foreach ($dir in $directories) {
    $size = Get-DirectorySize -path $dir.FullName
    $fileCount = Count-Files -path $dir.FullName
    $lastModified = (Get-Item $dir.FullName).LastWriteTime.ToString("yyyy-MM-dd HH:mm")
    $output += "\n{0,-30} {1,15} {2,10} {3}" -f $dir.Name, $size, $fileCount, $lastModified
}

# Get files in root
$files = Get-ChildItem -Path $rootDir -File | Sort-Object Name
$output += "\n\nFiles in Root Directory:"
$output += "\n" + ("-" * 80)
$output += "\n{0,-30} {1,15} {2}" -f "File", "Size", "Last Modified"
$output += "\n" + ("-" * 80)

foreach ($file in $files) {
    $size = "{0:N2} KB" -f ($file.Length / 1KB)
    $lastModified = $file.LastWriteTime.ToString("yyyy-MM-dd HH:mm")
    $output += "\n{0,-30} {1,15} {2}" -f $file.Name, $size, $lastModified
}

# Save to file
$output | Out-File -FilePath $outputFile -Encoding utf8

# Display summary
Write-Host "\nDirectory listing saved to: $outputFile"
Write-Host "Total directories: $($directories.Count)"
Write-Host "Total files in root: $($files.Count)"
