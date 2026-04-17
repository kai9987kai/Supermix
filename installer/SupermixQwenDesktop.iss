#ifndef MyAppName
  #define MyAppName "Supermix Qwen Desktop"
#endif
#ifndef MyAppExeName
  #define MyAppExeName "SupermixQwenDesktop.exe"
#endif
#ifndef MyAppVersion
  #define MyAppVersion "2026.03.06"
#endif
#ifndef MySourceDir
  #define MySourceDir "..\dist\SupermixQwenDesktop"
#endif
#ifndef MyOutputDir
  #define MyOutputDir "..\dist\installer"
#endif
#ifndef MySetupBaseName
  #define MySetupBaseName "SupermixQwenDesktopSetup"
#endif

[Setup]
AppId={{BCF0A53D-392D-49D8-A167-6B2F33005808}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher=Supermix
DefaultDirName={autopf}\Supermix Qwen Desktop
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
UninstallDisplayIcon={app}\{#MyAppExeName}
SetupIconFile=..\assets\supermix_qwen_icon.ico
WizardStyle=modern
WizardImageFile=..\assets\supermix_qwen_installer_wizard.bmp
WizardSmallImageFile=..\assets\supermix_qwen_installer_small.bmp
Compression=lzma2/max
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
ChangesAssociations=no
OutputDir={#MyOutputDir}
OutputBaseFilename={#MySetupBaseName}
SetupLogging=yes
InfoAfterFile=postinstall_notes.txt

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"; Flags: unchecked

[Files]
Source: "{#MySourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: "*.log,*.tmp,*.pyc,__pycache__"

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent runasoriginaluser

[Code]
function IsPythonOnPath(): Boolean;
var
  ResultCode: Integer;
begin
  Result := Exec(ExpandConstant('{cmd}'), '/C where python', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
  Result := Result and (ResultCode = 0);
end;

function InitializeSetup(): Boolean;
begin
  if not IsPythonOnPath() then
  begin
    MsgBox(
      'Python was not found on PATH. The launcher can still be installed, but it will not start the local model server until Python is installed and available as "python".',
      mbInformation,
      MB_OK
    );
  end;
  Result := True;
end;
