import os
import sys
import pathlib
# ensure backend package is importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / 'backend'))

# Ensure registry uses a temporary DB and models dir for this test
def test_registry_roundtrip(tmp_path, monkeypatch):
    tmp_db = tmp_path / 'test_registry.db'
    tmp_models = tmp_path / 'models'
    os.environ['PREDICTIFLOW_REGISTRY_DB'] = str(tmp_db)
    os.environ['PREDICTIFLOW_MODELS_DIR'] = str(tmp_models)

    # Import after env override so registry picks up the paths
    from app.registry import register_model, list_models, get_model, get_model_file_path

    # create a fake model file content
    content = b"dummy model"
    model_name = "test-model"
    filename = "model.bin"
    # register
    model_id = register_model(model_name, filename, content, metadata={"test": True})
    assert isinstance(model_id, int)
    models = list_models()
    assert any(m['id'] == model_id for m in models)
    m = get_model(model_id)
    assert m['name'] == model_name
    p = get_model_file_path(m['filename'])
    assert p is not None and os.path.exists(p)

    # cleanup file
    try:
        os.remove(p)
    except Exception:
        pass
