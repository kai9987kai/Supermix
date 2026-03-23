print('HAS_DATA_FILES', 'DATA_FILES' in globals())
if 'DATA_FILES' in globals():
    print('DATA_FILES_COUNT', len(DATA_FILES))
    for path in DATA_FILES:
        print('DATA_FILE', path)
print('HAS_TRAIN_CMD', 'TRAIN_CMD' in globals())
print('HAS_OUT_LOG', 'OUT_LOG' in globals())
print('HAS_WORKSPACE_DIR', 'WORKSPACE_DIR' in globals(), WORKSPACE_DIR if 'WORKSPACE_DIR' in globals() else None)
print('HAS_TRAINING_INPUT_DIR', 'TRAINING_INPUT_DIR' in globals(), TRAINING_INPUT_DIR if 'TRAINING_INPUT_DIR' in globals() else None)
