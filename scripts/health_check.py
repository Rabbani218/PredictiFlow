"""Simple health-check script for PredictiFlow services.

Checks:
- Backend /health endpoint (default http://localhost:8000/health)
- Frontend Streamlit port (default http://localhost:8501/)

Usage:
    python scripts/health_check.py --backend http://localhost:8000 --streamlit http://localhost:8501
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
import urllib.error
from urllib.parse import urljoin


def ping(url: str, timeout: float = 3.0) -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            code = r.getcode()
            return (200 <= code < 400, f"HTTP {code}")
    except urllib.error.HTTPError as e:
        return (False, f"HTTPError {e.code}")
    except urllib.error.URLError as e:
        return (False, f"URL Error: {e.reason}")
    except Exception as e:
        return (False, f"Error: {e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="http://localhost:8000", help="Base URL for backend")
    p.add_argument("--streamlit", default="http://localhost:8501", help="Base URL for streamlit frontend")
    args = p.parse_args()

    backend_health = urljoin(args.backend.rstrip("/"), "/health")
    ok_b, msg_b = ping(backend_health)
    print(f"Backend: {backend_health} -> {msg_b}")

    ok_s, msg_s = ping(args.streamlit)
    print(f"Streamlit: {args.streamlit} -> {msg_s}")

    if ok_b and ok_s:
        print("STATUS: OK")
        sys.exit(0)
    else:
        print("STATUS: UNHEALTHY")
        sys.exit(2)


if __name__ == '__main__':
    main()
