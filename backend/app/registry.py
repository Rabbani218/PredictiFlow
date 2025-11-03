import sqlite3
from pathlib import Path
import json
from datetime import datetime, timezone
import threading
import os
from typing import Union

# Allow overriding the registry DB path and models directory via environment variables
ROOT = Path(__file__).resolve().parents[2]
_env_db = os.environ.get('PREDICTIFLOW_REGISTRY_DB')
_env_models = os.environ.get('PREDICTIFLOW_MODELS_DIR')

DB_PATH = Path(_env_db) if _env_db else (ROOT / 'backend_registry.db')
MODELS_DIR = Path(_env_models) if _env_models else (ROOT / 'models')
MODELS_DIR.mkdir(parents=True, exist_ok=True)
_lock = threading.Lock()


def set_registry_paths(db_path: Union[str, Path], models_dir: Union[str, Path]):
    """Programmatically override registry DB path and models directory (useful for tests)."""
    global DB_PATH, MODELS_DIR
    DB_PATH = Path(db_path)
    MODELS_DIR = Path(models_dir)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _get_conn():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_registry():
    with _lock:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute(
            '''
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                filename TEXT,
                metadata TEXT,
                uploaded_at TEXT
            )
            '''
        )
        conn.commit()
        conn.close()


def register_model(name: str, filename: str, content: bytes, metadata: dict | None = None) -> int:
    """Save model file and register metadata. Returns model id."""
    init_registry()
    # use timezone-aware UTC timestamps
    now = datetime.now(timezone.utc)
    timestamp = now.isoformat()
    safe_name = f"{now.strftime('%Y%m%d%H%M%S%f')}_{filename}"
    out_path = MODELS_DIR / safe_name
    with open(out_path, 'wb') as f:
        f.write(content)

    meta_json = json.dumps(metadata or {})
    with _lock:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute("INSERT INTO models (name, filename, metadata, uploaded_at) VALUES (?, ?, ?, ?)",
                    (name, safe_name, meta_json, timestamp))
        model_id = cur.lastrowid
        conn.commit()
        conn.close()
    return model_id


def list_models(limit: int = 100):
    init_registry()
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, name, filename, metadata, uploaded_at FROM models ORDER BY uploaded_at DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    out = []
    for r in rows:
        out.append({
            'id': r['id'],
            'name': r['name'],
            'filename': r['filename'],
            'metadata': json.loads(r['metadata'] or '{}'),
            'uploaded_at': r['uploaded_at']
        })
    return out


def get_model(model_id: int):
    init_registry()
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, name, filename, metadata, uploaded_at FROM models WHERE id = ?", (model_id,))
    r = cur.fetchone()
    conn.close()
    if not r:
        return None
    return {
        'id': r['id'],
        'name': r['name'],
        'filename': r['filename'],
        'metadata': json.loads(r['metadata'] or '{}'),
        'uploaded_at': r['uploaded_at']
    }


def get_model_file_path(filename: str):
    p = MODELS_DIR / filename
    if p.exists():
        return p
    return None
