import os
import socket
import subprocess
import time
from dotenv import load_dotenv


load_dotenv()


def check_pgvector_running(start_if_missing: bool = False) -> bool:
    """
    Check if PGVector/Postgres is running using PG_HOST and PG_PORT from .env.

    Args:
        start_if_missing (bool): If True, attempts to start PGVector via docker-compose.

    Returns:
        bool: True if PGVector is running.

    Raises:
        RuntimeError: If PGVector is not running and start_if_missing is False,
                      or if starting via docker-compose fails.
    """
    host = os.environ.get("PG_HOST")
    port = os.environ.get("PG_PORT")

    if not host or not port:
        raise RuntimeError("PG_HOST and/or PG_PORT not set in .env")

    port = int(port)

    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.settimeout(2)
        sock.connect((host, port))
        sock.close()
        print(f"PGVector is running at {host}:{port}")
        return True
    except Exception:
        if start_if_missing:
            print("PGVector not running. Attempting to start via docker-compose...")
            ret = subprocess.run(["docker", "compose", "up", "-d", "pgvector"])
            if ret.returncode != 0:
                raise RuntimeError("Failed to start PGVector via docker-compose")
            # Wait a few seconds for healthcheck
            print("Waiting 5 seconds for PGVector to become ready...")
            time.sleep(5)
            return True
        else:
            raise RuntimeError(f"PGVector not running at {host}:{port}. "
                               "Set start_if_missing=True to auto-start.")

    return False
