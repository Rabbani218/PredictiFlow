import os
import sys
import pathlib

# Ensure app uses a temporary registry during tests
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / 'backend'))

def _make_client_with_tmp_registry(tmp_path):
    tmp_db = tmp_path / 'test_registry.db'
    tmp_models = tmp_path / 'models'
    os.environ['PREDICTIFLOW_REGISTRY_DB'] = str(tmp_db)
    os.environ['PREDICTIFLOW_MODELS_DIR'] = str(tmp_models)
    # Import app after setting env so registry is configured with tmp paths
    from app.main import app
    from fastapi.testclient import TestClient
    return TestClient(app)


def test_models_list_empty(tmp_path):
    client = _make_client_with_tmp_registry(tmp_path)
    r = client.get('/models')
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_upload_and_download(tmp_path):
    client = _make_client_with_tmp_registry(tmp_path)
    # create a small dummy file
    p = tmp_path / 'dummy.bin'
    p.write_bytes(b'abc')
    with open(p, 'rb') as fh:
        files = {'file': ('dummy.bin', fh, 'application/octet-stream')}
        r = client.post('/models', files=files)
    assert r.status_code == 200
    body = r.json()
    assert 'id' in body
    model_id = body['id']
    # download
    r2 = client.get(f'/models/{model_id}/download')
    assert r2.status_code == 200
    assert r2.content == b'abc'
