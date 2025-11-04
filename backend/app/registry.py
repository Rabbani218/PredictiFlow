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
    if str(DB_PATH) != ':memory:':
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _create_tables(conn):
    """Create tables in the database. Safe to call multiple times."""
    cur = conn.cursor()
    cur.execute(
        '''
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            filename TEXT NOT NULL UNIQUE,
            metadata TEXT,
            uploaded_at TEXT NOT NULL
        )
        '''
    )
    cur.execute("SELECT COUNT(*) FROM models")  # verify table exists


def _get_conn():
    """Get a SQLite connection with row factory enabled and tables created."""
    # ensure parent directory exists if using file-based DB
    if str(DB_PATH) != ':memory:':
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Use immediate mode to reduce locking issues, and enable row factory
    conn = sqlite3.connect(str(DB_PATH), isolation_level=None, 
                         uri=True if str(DB_PATH) == ':memory:' else False)
    conn.row_factory = sqlite3.Row
    
    # Always ensure tables exist when getting a new connection
    try:
        _create_tables(conn)
    except Exception as e:
        conn.close()
        raise RuntimeError(f"Failed to create tables: {e}")
    
    return conn


def init_registry(force_create=False):
    """Initialize registry database with required tables.
    
    Args:
        force_create: If True, drop and recreate the table. Use with caution.
    """
    with _lock:
        conn = None
        try:
            conn = sqlite3.connect(str(DB_PATH), isolation_level=None)
            cur = conn.cursor()
            
            if force_create:
                cur.execute("DROP TABLE IF EXISTS models")
            
            _create_tables(conn)
            
        except Exception as e:
            print(f"Error initializing registry at {DB_PATH}: {e}")
            raise
        finally:
            if conn:
                conn.close()


def register_model(name: str, filename: str, content: bytes, metadata: dict | None = None) -> int:
    """Save model file and register metadata. Returns model id."""
    init_registry()
    if metadata is None:
        metadata = {}
    # use timezone-aware UTC timestamps
    now = datetime.now(timezone.utc)
    timestamp = now.isoformat()
    safe_name = f"{now.strftime('%Y%m%d%H%M%S%f')}_{filename}"
    out_path = MODELS_DIR / safe_name
    
    # Ensure models directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Add original filename to metadata
    metadata['original_filename'] = filename
    meta_json = json.dumps(metadata)
    
    with _lock:
        conn = _get_conn()  # already in autocommit mode
        try:
            # First write the file
            with open(out_path, 'wb') as f:
                f.write(content)
            
            # Then update the database
            cur = conn.cursor()
            cur.execute("BEGIN IMMEDIATE")
            cur.execute(
                "INSERT INTO models (name, filename, metadata, uploaded_at) VALUES (?, ?, ?, ?)",
                (name, safe_name, meta_json, timestamp)
            )
            model_id = cur.lastrowid
            conn.commit()
            return model_id
        except Exception as e:
            cur.execute("ROLLBACK")
            # If db insert fails, clean up the saved file
            try:
                os.unlink(out_path)
            except Exception:
                pass
            raise
        finally:
            conn.close()


def list_models(limit: int = 100):
    """List all models in the registry, sorted by upload time (newest first)."""
    init_registry()
    with _lock:
        conn = _get_conn()
        try:
            cur = conn.cursor()
            cur.execute("SELECT id, name, filename, metadata, uploaded_at FROM models ORDER BY uploaded_at DESC LIMIT ?", (limit,))
            rows = cur.fetchall()
            
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
        except Exception:
            raise
        finally:
            conn.close()


def get_model(model_id: int):
    """Get a single model by ID."""
    init_registry()
    with _lock:
        conn = _get_conn()
        try:
            cur = conn.cursor()
            cur.execute("BEGIN IMMEDIATE")
            cur.execute("SELECT id, name, filename, metadata, uploaded_at FROM models WHERE id = ?", (model_id,))
            r = cur.fetchone()
            cur.execute("COMMIT")
            if r is None:
                return None
            return {
                'id': r['id'],
                'name': r['name'],
                'filename': r['filename'],
                'metadata': json.loads(r['metadata'] or '{}'),
                'uploaded_at': r['uploaded_at']
            }
        except Exception:
            cur.execute("ROLLBACK")
            raise
        finally:
            conn.close()


def get_model_file_path(filename: str):
    """Get absolute path to a model file."""
    p = MODELS_DIR / filename
    if p.is_file():  # Only return if it's a regular file, not directory
        return p
    return None
