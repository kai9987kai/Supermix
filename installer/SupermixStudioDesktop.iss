#ifndef MyAppName
  #define MyAppName "Supermix Studio"
#endif
#ifndef MyAppExeName
  #define MyAppExeName "SupermixStudioDesktop.exe"
#endif
#ifndef MyAppVersion
  #define MyAppVersion "2026.03.27"
#endif
#ifndef MySourceDir
  #define MySourceDir "..\dist\SupermixStudioDesktop"
#endif
#ifndef MyOutputDir
  #define MyOutputDir "..\dist\installer"
#endif
#ifndef MySetupBaseName
  #define MySetupBaseName "SupermixStudioDesktopSetup"
#endif

[Setup]
AppId={{6AB55323-1FBB-4C1D-BF0E-1155845F1F76}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher=Supermix
DefaultDirName={autopf}\Supermix Studio
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
InfoAfterFile=postinstall_notes_studio.txt

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
