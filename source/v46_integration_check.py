import sys
from pathlib import Path

# Add source to path
sys.path.append(str(Path(__file__).parent))

try:
    from multimodel_catalog import discover_model_records, MODEL_SPECS
    from multimodel_runtime import UnifiedModelManager, ChatResult
    print("Imports successful.")

    # 1. Check Catalog
    v46_spec = next((s for s in MODEL_SPECS if s.key == "omni_collective_v46"), None)
    if v46_spec:
        print(f"V46 Spec found: {v46_spec.label}")
    else:
        print("V46 Spec NOT found in MODEL_SPECS.")

    # 2. Check Backend instantiation (mocking weights if needed, but here we just check mapping)
    # We can't easily instantiate without weights, but we can check the kind mapping in manager
    manager = UnifiedModelManager(records=MODEL_SPECS, extraction_root=Path("tmp/ext"), generated_dir=Path("tmp/gen"))
    
    # Check if 'omni_collective_v46' kind is handled
    # (Checking if it crashes or if the class exists in the module)
    from multimodel_runtime import OmniCollectiveV46Backend
    print("OmniCollectiveV46Backend class exists.")

except Exception as e:
    print(f"Error during verification: {e}")
    import traceback
    traceback.print_exc()
