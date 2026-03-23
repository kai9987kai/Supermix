import shutil
import subprocess
import sys

from importlib.metadata import PackageNotFoundError, version


def run(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print('+', ' '.join(cmd))
    try:
        subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None, env=env)
    except subprocess.CalledProcessError as exc:
        joined = ' '.join(cmd)
        raise RuntimeError(f'Command failed with exit code {exc.returncode}: {joined}') from exc


def find_source_snapshot_dir(root: Path) -> Path | None:
    candidates = [
        root / 'source',
        root,
    ]
    for candidate in candidates:
        if (candidate / 'qwen_supermix_pipeline.py').exists():
            return candidate
    return None


def extract_archives(input_dir: Path, dest_dir: Path) -> list[Path]:
    extracted: list[Path] = []
    if not input_dir.exists():
        return extracted
    for archive_path in sorted(input_dir.glob('*.zip')):
        print(f'Extracting archive: {archive_path} -> {dest_dir}')
        dest_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(dest_dir)
        extracted.append(archive_path)
    return extracted


if WORKSPACE_DIR.exists():
    print(f'Removing existing workspace at {WORKSPACE_DIR} to avoid stale state...')
    shutil.rmtree(WORKSPACE_DIR)

source_snapshot_dir = find_source_snapshot_dir(SOURCE_INPUT_DIR)
if source_snapshot_dir is not None:
    print(f'Using Kaggle source dataset: {source_snapshot_dir}')
    (WORKSPACE_DIR / 'source').parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_snapshot_dir, WORKSPACE_DIR / 'source', dirs_exist_ok=True)
else:
    source_archives = extract_archives(SOURCE_INPUT_DIR, WORKSPACE_DIR)
    extracted_source_dir = find_source_snapshot_dir(WORKSPACE_DIR)
    if extracted_source_dir is not None:
        if extracted_source_dir == WORKSPACE_DIR:
            raise RuntimeError('Source archive must contain a top-level source/ directory.')
        print(f'Using extracted Kaggle source archive(s): {source_archives}')
    else:
        if not ALLOW_GIT_CLONE_FALLBACK:
            raise FileNotFoundError(
                f'Missing source dataset at {SOURCE_INPUT_DIR}. Attach a Kaggle dataset named {SOURCE_DATASET_SLUG} or enable ALLOW_GIT_CLONE_FALLBACK.'
            )
        repo_name = REPO_URL.rstrip('/').rsplit('/', 1)[-1].removesuffix('.git')
        archive_url = f'https://codeload.github.com/kai9987kai/{repo_name}/zip/refs/heads/{BRANCH}'
        archive_path = KAGGLE_WORKING_ROOT / f'{repo_name}_{BRANCH}_source.zip'
        extract_root = KAGGLE_WORKING_ROOT / '_repo_source_extract'
        if archive_path.exists():
            archive_path.unlink()
        if extract_root.exists():
            shutil.rmtree(extract_root)
        print(f'Using archive fallback instead of git clone: {archive_url}')
        urllib.request.urlretrieve(archive_url, archive_path)
        with zipfile.ZipFile(archive_path) as zf:
            source_prefix = f'{repo_name}-{BRANCH}/source/'
            members = [name for name in zf.namelist() if name.startswith(source_prefix)]
            if not members:
                raise RuntimeError(f'Archive fallback did not contain {source_prefix}')
            zf.extractall(extract_root, members)
        extracted_source_dir = extract_root / f'{repo_name}-{BRANCH}' / 'source'
        shutil.copytree(extracted_source_dir, WORKSPACE_DIR / 'source', dirs_exist_ok=True)
        print(f'Using downloaded source archive fallback: {extracted_source_dir}')

run([sys.executable, '-m', 'pip', 'install', '-q', '--upgrade', 'pip', 'setuptools', 'wheel'])
run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'source/requirements_train_build.txt'], cwd=WORKSPACE_DIR)

extra_packages = [
    'transformers',
    'peft',
    'accelerate',
    'safetensors',
    'matplotlib',
    'tokenizers',
    'huggingface_hub',
    'sentencepiece',
    'tqdm',
]
if PINNED_VERSIONS:
    pinned = [f'{name}=={ver}' for name, ver in PINNED_VERSIONS.items()]
    run([sys.executable, '-m', 'pip', 'install', '-q', *pinned], cwd=WORKSPACE_DIR)
else:
    run([sys.executable, '-m', 'pip', 'install', '-q', *extra_packages], cwd=WORKSPACE_DIR)

for package_name in ['torch', 'transformers', 'peft', 'accelerate', 'safetensors']:
    try:
        print(package_name, version(package_name))
    except PackageNotFoundError:
        print(package_name, '<missing>')
