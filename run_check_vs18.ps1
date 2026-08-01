$vcDir = "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC"
$vsDir = "C:\Program Files\Microsoft Visual Studio\18\Insiders"
$msvcVer = "14.44.35207"
$kitVer = "10.0.26100.0"
$kitDir = "C:\Program Files (x86)\Windows Kits\10"

$clPath = "$vcDir\Tools\MSVC\$msvcVer\bin\Hostx64\x64\cl.exe"
$linkPath = "$vcDir\Tools\MSVC\$msvcVer\bin\Hostx64\x64\link.exe"

$env:CC = $clPath
$env:CXX = $clPath
$env:VCINSTALLDIR = $vcDir
$env:VSINSTALLDIR = $vsDir
$env:WindowsSdkDir = $kitDir
$env:INCLUDE = "$vcDir\Tools\MSVC\$msvcVer\include;$kitDir\Include\$kitVer\shared;$kitDir\Include\$kitVer\um;$kitDir\Include\$kitVer\ucrt"
$env:LIB = "$vcDir\Tools\MSVC\$msvcVer\lib\x64;$kitDir\Lib\$kitVer\um\x64;$kitDir\Lib\$kitVer\ucrt\x64"
$env:LIBPATH = "$vcDir\Tools\MSVC\$msvcVer\lib\x64;$kitDir\Lib\$kitVer\um\x64"
$env:Path = "$vcDir\Tools\MSVC\$msvcVer\bin\Hostx64\x64;$kitDir\bin\$kitVer\x64;$env:Path"

cargo check --all-targets --all-features 2>&1
