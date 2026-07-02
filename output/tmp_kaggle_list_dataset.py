from pathlib import Path
root = TRAINING_INPUT_DIR
print('TRAINING_INPUT_DIR', root)
if not root.exists():
    print('MISSING_INPUT_DIR')
else:
    paths = sorted(root.rglob('*'))
    print('TOTAL_PATHS', len(paths))
    for path in paths[:120]:
        kind = 'DIR ' if path.is_dir() else 'FILE'
        print(kind, path.relative_to(root))
