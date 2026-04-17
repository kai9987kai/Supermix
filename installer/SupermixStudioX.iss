#ifndef MyAppName
  #define MyAppName "Supermix Studio X V48"
#endif
#ifndef MyAppVersion
  #define MyAppVersion "v48.0.0"
#endif
#ifndef MyAppExeName
  #define MyAppExeName "SupermixStudioX.exe"
#endif
#ifndef MySourceDir
  #define MySourceDir "..\dist\SupermixStudioX"
#endif
#ifndef MyOutputDir
  #define MyOutputDir "..\dist\installer"
#endif
#ifndef MySetupBaseName
  #define MySetupBaseName "SupermixStudioX_V48_Setup"
#endif

[Setup]
AppId={{A1B2C3D4-E5F6-4A7B-8C9D-0E1F2A3B4C5D}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher=Supermix AI Research
AppPublisherURL=https://github.com/supermix-ai
AppSupportURL=https://github.com/supermix-ai
AppUpdatesURL=https://github.com/supermix-ai
DefaultDirName={autopf}\Supermix Studio X
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
DiskSpanning=no
InfoAfterFile=postinstall_notes_studio_x.txt

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; Flags: unchecked
Name: "quicklaunch"; Description: "Add to Quick &Launch"; Flags: unchecked

[InstallDelete]
Type: filesandordirs; Name: "{app}\_internal\bundled_models"
Type: filesandordirs; Name: "{app}\_internal\bundled_models_v46"
Type: filesandordirs; Name: "{app}\_internal\bundled_models_v47"
Type: filesandordirs; Name: "{app}\_internal\bundled_models_v48"
Type: filesandordirs; Name: "{app}\_internal\bundled_base_model"

[Files]
Source: "{#MySourceDir}\*"; DestDir: "{app}"; \
  Flags: ignoreversion recursesubdirs createallsubdirs; \
  Excludes: "*.log,*.tmp,*.pyc,__pycache__,*.pid"

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; \
  WorkingDir: "{app}"; Comment: "Supermix Studio X - V48 Frontier"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; \
  WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; \
  Description: "Launch {#MyAppName}"; \
  Flags: nowait postinstall skipifsilent runasoriginaluser

[UninstallDelete]
Type: filesandordirs; Name: "{app}\tmp"
Type: filesandordirs; Name: "{app}\run_state_studio"
