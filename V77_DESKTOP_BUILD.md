# v77 — Supermix Chat as a desktop application

The chat interface, packaged as a Windows application with v74 inside it. No
Python, no command line, no download step, and no network access at all.

```
Supermix Chat
  what is 47 times 6
    asked as: What is 47 x 6?  (multiplication)
    40 x 6 = 240, 7 x 6 = 42, total 282
    CORRECT — 282
```

## What was built

| artifact | size |
|---|---|
| `dist/SupermixChatDesktop/` | 1,735 MB |
| `SupermixChatDesktop.exe` | 62 MB |
| bundled model (`_internal/model/supermix_v74.pt`) | 33 MB |
| `dist/installer/SupermixChat-74.0.0.zip` | 560 MB |

`source/supermix_chat_desktop_app.py` is deliberately thin. The Flask server is
**unmodified**, so everything true of the web interface is true here: the
prompt normaliser, the independent answer check, bounded concurrency. The
wrapper only resolves the bundled checkpoint, starts that same server on a free
port, and shows it in a window.

Three packaging decisions worth stating:

* **The model ships inside the executable**, unpacked by PyInstaller to
  `sys._MEIPASS`. `bundled_checkpoint()` looks there first and falls back to
  the repository layout for development. A build that quietly found the
  developer's `output/` directory would work on this machine and nowhere else,
  so the spec *raises* if no model is staged rather than producing a
  model-less application that reports success.
* **Training state is stripped.** The v74 checkpoint is 103.5 MB, of which
  ~69 MB is AdamW moments and the LR schedule. Those resume training and are
  dead weight in an inference application. Stripped: 33 MB, loads identically.
* **The recall corpus is not bundled.** At 101.6 MB against the model's 33 MB
  it would nearly double the payload to power one diagnostic — the verbatim
  meter. The answer check, which is what makes a reply trustworthy, needs no
  corpus and is always on.

Size is dominated by torch: 448 MB installed, and its DLLs expand further.
That is the floor for shipping a PyTorch model as a standalone Windows app.

## The build failed first, and the way it failed matters

The first build completed successfully and produced a 1.8 GB application that
died instantly on launch:

```
ImportError: cannot import name 'distributions' from partially
initialized module 'torch'
```

The cause was mine: `torch.distributions` was in the spec's `excludes`, to save
space. torch's own `__init__` imports it, so excluding it broke the package
outright.

**PyInstaller reported success.** Nothing in 1.3 million lines of build log
indicated a problem, and the executable was windowed (`console=False`) so it
also produced no visible error — it just vanished. The failure was only
findable by running the exe with stderr redirected to a file.

`test_chat_desktop_packaging.py` now parses the spec and fails if any exclude
contains a dot. Only whole top-level packages may be excluded; if a package
ships at all, PyInstaller decides which parts of it go.

A second failure was also self-inflicted: the COLLECT stage died with
`PermissionError: [WinError 32]` because an earlier command had left a shell's
working directory *inside* `dist/SupermixChatDesktop`, which holds a lock on
it. Build from the repository root.

## The installer

`build_chat_desktop_installer.ps1` prefers Inno Setup 6 and compiles
`installer/SupermixChatDesktop.iss` into a single `SupermixChatSetup.exe`.

**Inno Setup is not installed on this machine**, so it took the fallback path
and produced a zip plus `Install-SupermixChat.ps1`. That fallback is a real
installer, not a copy script:

* installs per-user to `%LOCALAPPDATA%\Programs\Supermix Chat` — no
  administrator rights, nothing written outside the user profile
* creates a Start Menu entry and an optional desktop shortcut
* registers in Add/Remove Programs (verified: *Supermix Chat 74.0.0*,
  1,735 MB)
* writes a standalone `Uninstall-SupermixChat.ps1` that does not depend on the
  installer surviving, and which **refuses to delete a directory that does not
  contain the application** rather than trusting a path

Compression uses `ZipFile::CreateFromDirectory` rather than `Compress-Archive`,
which is slow over this many files and unreliable near the 2 GB mark — where a
1.8 GB torch application sits.

## Verified end to end

Not "it built" — installed and run:

| check | result |
|---|---|
| exe launches and serves | yes |
| `what is 47 times 6` | CORRECT — 282 |
| `what comes next: 5, 12, 19, 26` | CORRECT — 33 |
| `what is 20% of 150 then add 12` | CORRECT — 42 |
| `what is 15 percent of 240` | **WRONG — said 26.0, expected 36.0** |
| `hello` | NOT CHECKED |
| Start Menu + desktop shortcuts | present, correct target |
| Add/Remove Programs | registered |

The `percent` row is the model being wrong and the app **saying so**. That is
the intended behaviour: `percent` scores 0.75, and a wrong answer surfaced as
WRONG is worth more than confident arithmetic the reader has to check.

## What this application is not

It is not a chat model. It answers "hello" fluently and every conversational
reply is reproduced verbatim from training data. The post-install notes say so
in those words, because a user meeting an 8.6M-parameter model through a
polished window will otherwise assume otherwise.
