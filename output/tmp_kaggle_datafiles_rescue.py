from pathlib import Path
import zipfile

DATASET_FILE_NAMES = [
    'conversation_data.quality_anchor_v2.jsonl',
    'conversation_data.coding_knowledge_2026_02_19.jsonl',
    'conversation_data.world_events_2026_02_19.jsonl',
    'conversation_data.supermix_plus_v27_500k.jsonl',
    'conversation_data.mega_reasoning_creative_v25_75582.jsonl',
    'conversation_data.mega_creative_250k_v2.jsonl',
]

def extract_training_archives_recursive() -> None:
    if not TRAINING_INPUT_DIR.exists():
        return
    dest_dir = WORKSPACE_DIR / 'datasets'
    dest_dir.mkdir(parents=True, exist_ok=True)
    for archive_path in sorted(TRAINING_INPUT_DIR.rglob('*.zip')):
        print(f'Extracting training archive: {archive_path} -> {dest_dir}')
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(dest_dir)

def resolve_dataset_files_recursive() -> list[str]:
    resolved: list[str] = []
    missing: list[str] = []
    search_roots = [
        TRAINING_INPUT_DIR,
        TRAINING_INPUT_DIR / 'datasets',
        WORKSPACE_DIR / 'datasets',
        WORKSPACE_DIR,
        KAGGLE_WORKING_ROOT,
    ]
    for name in DATASET_FILE_NAMES:
        candidates: list[Path] = []
        for root in search_roots:
            if not root.exists():
                continue
            direct = [root / name, root / 'datasets' / name]
            for candidate in direct:
                if candidate.exists() and candidate.is_file():
                    candidates.append(candidate)
            if not candidates:
                try:
                    for hit in sorted(root.rglob(name)):
                        if hit.is_file():
                            candidates.append(hit)
                except Exception:
                    pass
        chosen = next((candidate for candidate in candidates if candidate.exists()), None)
        if chosen is None:
            missing.append(name)
        else:
            resolved.append(str(chosen.resolve()))
    if missing:
        raise FileNotFoundError('Missing training JSONL files: ' + ', '.join(missing))
    return resolved

extract_training_archives_recursive()
DATA_FILES = resolve_dataset_files_recursive()
print('Resolved dataset files:')
for dataset_path in DATA_FILES:
    print('-', dataset_path)
