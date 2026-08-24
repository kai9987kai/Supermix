; Inno Setup script for Supermix Chat (v74 built in).
;
; Compile with:  ISCC.exe installer\SupermixChatDesktop.iss
; Build the app first:  .\build_chat_desktop_exe.ps1
;
; Installs per-user under %LOCALAPPDATA%\Programs so no administrator rights
; are required. The model ships inside the application; there is no download
; step and the app never contacts the network.

#ifndef MyAppName
  #define MyAppName "Supermix Chat"
#endif
#ifndef MyAppExeName
  #define MyAppExeName "SupermixChatDesktop.exe"
#endif
#ifndef MyAppVersion
  #define MyAppVersion "74.0.0"
#endif
#ifndef MySourceDir
  #define MySourceDir "..\dist\SupermixChatDesktop"
#endif
#ifndef MyOutputDir
  #define MyOutputDir "..\dist\installer"
#endif
#ifndef MySetupBaseName
  #define MySetupBaseName "SupermixChatSetup"
#endif

[Setup]
AppId={{B7F4C2A1-9E3D-4A58-8C61-2D7E5F0A9B34}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher=Supermix
DefaultDirName={localappdata}\Programs\Supermix Chat
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
UninstallDisplayIcon={app}\{#MyAppExeName}
SetupIconFile=..\assets\supermix_qwen_icon.ico
WizardStyle=modern
Compression=lzma2/max
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#MyOutputDir}
OutputBaseFilename={#MySetupBaseName}
SetupLogging=yes
InfoAfterFile=postinstall_notes_chat.txt

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"; Flags: unchecked

[InstallDelete]
; The checkpoint is shipped as bundled data. Remove the previous one on
; upgrade so an older model cannot survive alongside the new one and be
; loaded by a stale path.
Type: filesandordirs; Name: "{app}\_internal\model"

[Files]
Source: "{#MySourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: "*.log,*.tmp,*.pyc,__pycache__"

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent runasoriginaluser
