import sqlite3
import json
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List

DB_PATH = Path(__file__).resolve().parents[3] / 'data' / 'jobs.db'
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def _get_conn():
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        '''
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            status TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            finished_at TEXT,
            payload TEXT,
            result TEXT,
            error TEXT
        )
        '''
    )
    conn.commit()
    conn.close()


def create_job(payload: Dict[str, Any]) -> int:
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("INSERT INTO jobs (status, payload) VALUES (?, ?)", ("pending", json.dumps(payload)))
    job_id = cur.lastrowid
    conn.commit()
    conn.close()
    return job_id


def update_job_result(job_id: int, result: Optional[Dict[str, Any]] = None, error: Optional[str] = None):
    conn = _get_conn()
    cur = conn.cursor()
    if error:
        cur.execute("UPDATE jobs SET status=?, error=?, finished_at=datetime('now') WHERE id=?", ("error", error, job_id))
    else:
        cur.execute("UPDATE jobs SET status=?, result=?, finished_at=datetime('now') WHERE id=?", ("done", json.dumps(result), job_id))
    conn.commit()
    conn.close()


def set_job_running(job_id: int):
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("UPDATE jobs SET status=? WHERE id=?", ("running", job_id))
    conn.commit()
    conn.close()


def get_job(job_id: int) -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM jobs WHERE id=?", (job_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return {
        'id': row['id'],
        'status': row['status'],
        'created_at': row['created_at'],
        'finished_at': row['finished_at'],
        'payload': json.loads(row['payload']) if row['payload'] else None,
        'result': json.loads(row['result']) if row['result'] else None,
        'error': row['error']
    }


def list_jobs(limit: int = 50) -> List[Dict[str, Any]]:
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM jobs ORDER BY id DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    out = []
    for row in rows:
        out.append({
            'id': row['id'],
            'status': row['status'],
            'created_at': row['created_at'],
            'finished_at': row['finished_at']
        })
    return out


# initialize DB on import in a thread-safe manner
_init_lock = threading.Lock()
with _init_lock:
    init_db()
