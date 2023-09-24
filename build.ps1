function Build-Project {
    param (
        [int]$clientValue,
        [string]$outputName,
        [string]$testName
    )

    New-Item -ItemType Directory -Force -Path build | Out-Null
    Set-Location build

    if (-not $testName) {
        cmake -DCLIENT_VALUE=$clientValue -DUSING_WINDOWS=1 ..
    } else {
        cmake -DCLIENT_VALUE=$clientValue -DUSING_WINDOWS=1 -DTEST_NAME=$testName ..
    }
    
    Copy-Item compile_commands.json ..
    make
    Move-Item "Hivemind.exe" $outputName
    Move-Item $outputName ..
    Write-Host "./$outputName built successfully"

    Set-Location ..
    Remove-Item -Recurse -Force build
}

if ($args.Count -lt 1) {
    Write-Host "Usage: $0 [node|test] [test_name]"
    exit 1
}

switch ($args[0]) {
    "node" {
        Build-Project -clientValue 1 -outputName "node.exe"
    }
    "test" {
        if (-not $args[1]) {
            Write-Host "Test name must be provided when using the 'test' option."
            exit 1
        }
        Build-Project -clientValue 1 -outputName "test.exe" -testName $args[1]
    }
    default {
        Write-Host "Unknown option: $($args[0])"
        Write-Host "Usage: $0 [node|test] [test_name]"
        exit 1
    }
}
