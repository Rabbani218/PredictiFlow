import os
import sys
import pathlib
# ensure backend package is importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / 'backend'))

# Ensure registry uses a temporary DB and models dir for this test
def test_registry_roundtrip(tmp_path, monkeypatch):
    tmp_db = tmp_path / 'test_registry.db'
    tmp_models = tmp_path / 'models'

    # Import module first
    from app.registry import (
        set_registry_paths, register_model, list_models, get_model, 
        get_model_file_path, init_registry, DB_PATH, MODELS_DIR
    )
    
    # Set paths and initialize
    set_registry_paths(tmp_db, tmp_models)
    init_registry(force_create=True)
    
    print(f"Using DB path: {DB_PATH}")
    print(f"Using models dir: {MODELS_DIR}")

    # create a fake model file content
    content = b"dummy model"
    model_name = "test-model"
    filename = "model.bin"
    # register
    model_id = register_model(model_name, filename, content, metadata={"test": True})
    assert isinstance(model_id, int)
    print(f"Registered model with ID: {model_id}")
    
    models = list_models()
    print(f"Listed models: {models}")
    assert models, "list_models() returned empty list"
    assert any(m['id'] == model_id for m in models), f"Model {model_id} not found in {models}"
    
    m = get_model(model_id)
    assert m is not None, "get_model() returned None"
    assert m['name'] == model_name
    
    p = get_model_file_path(m['filename'])
    assert p is not None, "get_model_file_path() returned None"
    assert os.path.exists(p), f"Model file {p} does not exist"
    
    # cleanup file
    try:
        os.unlink(p)
    except Exception as e:
        print(f"Failed to clean up file {p}: {e}")
