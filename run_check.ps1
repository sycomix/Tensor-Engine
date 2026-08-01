$env:CC = 'C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe'
$env:CXX = 'C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe'
$env:VCINSTALLDIR = 'C:\Program Files\Microsoft Visual Studio\18\Insiders\VC'
$env:VSINSTALLDIR = 'C:\Program Files\Microsoft Visual Studio\18\Insiders'
cargo check --all-targets --all-features 2>&1
