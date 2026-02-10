import socket
import os
import sys
import time
import logging
import subprocess
import mlflow
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logger = logging.getLogger(__name__)


def _is_server_running(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, int(port))) == 0


def _ensure_pgvector(log_file: str) -> None:
    host = os.environ.get("PG_HOST_", "localhost")
    port = int(os.environ.get("PG_PORT_", 6025))

    if _is_server_running(host, port):
        logger.info(f"PGVector running at {host}:{port}")
        return

    logger.info("Starting PGVector...")
    subprocess.run(
        ["docker", "compose", "up", "-d", "pgvector"],
        check=True,
        stdout=log_file,
        stderr=log_file,
    )

    for _ in range(90):
        if _is_server_running(host, port):
            logger.info("PGVector started.")
            return
        time.sleep(1)

    raise RuntimeError("PGVector failed to start.")


def _ensure_redis(log_file: str) -> None:
    host = os.environ.get("REDIS_HOST_CACHE", "localhost")
    port = int(os.environ.get("REDIS_PORT_CACHE", 6380))

    if _is_server_running(host, port):
        logger.info(f"Redis running at {host}:{port}")
        return

    logger.info("Starting Redis...")
    subprocess.run(
        ["docker", "compose", "up", "-d", "redis_cache"],
        check=True,
        stdout=log_file,
        stderr=log_file,
    )

    for _ in range(60):
        if _is_server_running(host, port):
            logger.info("Redis started.")
            return
        time.sleep(1)

    raise RuntimeError("Redis failed to start.")


def _ensure_qdrant(log_file: str) -> None:
    host = os.environ.get("QDRANT_HOST", "localhost")
    port = int(os.environ.get("QDRANT_PORT", 6333))

    if _is_server_running(host, port):
        logger.info(f"Qdrant running at {host}:{port}")
        return

    logger.info("Starting Qdrant...")
    subprocess.run(
        ["docker", "compose", "up", "-d", "qdrant"],
        check=True,
        stdout=log_file,
        stderr=log_file,
    )

    for _ in range(60):
        if _is_server_running(host, port):
            logger.info("Qdrant started.")
            return
        time.sleep(1)

    raise RuntimeError("Qdrant failed to start.")


def _ensure_langfuse(log_file: str) -> None:
    host = os.environ["LANGFUSE_HOST"]
    port = int(os.environ["LANGFUSE_PORT"])
    langfuse_wd_path = os.path.expanduser(os.environ["LANGFUSE_PATH"])

    if _is_server_running(host, port):
        logger.info(f"Langfuse running at {host}:{port}")
        return

    logger.info("Starting Langfuse.")
    subprocess.run(
        ["docker", "compose", "-f", "docker-compose.yml", "up", "-d"],
        check=True,
        stdout=log_file,
        stderr=log_file,
        cwd=langfuse_wd_path,
    )

    for _ in range(90):
        if _is_server_running(host, port):
            logger.info("Langfuse started.")
            return langfuse_wd_path
        time.sleep(1)

    raise RuntimeError("Langfuse failed to start.")


def _ensure_mlflow(cfg: dict, log_file: str) -> None:
    host = os.environ["MLFLOW_HOST"]
    port = os.environ["MLFLOW_PORT"]

    if _is_server_running(host, int(port)):
        logger.info(f"Mlflow running at {host}:{port}")
        return

    logger.info("Starting Mlflow.")

    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mlflow",
            "server",
            "--host",
            host,
            "--port",
            port,
            "--backend-store-uri",
            "sqlite:///mlflow/mlflow.db",
            "--default-artifact-root",
            "mlflow/artifacts",
        ],
        stdout=log_file,
        stderr=log_file,
    )

    for _ in range(60):
        if _is_server_running(host, port):
            mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
            mlflow.set_experiment(cfg["experiment"])
            logger.info("Mlflow started.")
            return
        time.sleep(1)

    raise RuntimeError("Mlflow failed to start.")


def _ensure_llamacpp() -> None:
    host = os.environ["MODEL_HOST"]
    port = int(os.environ["MODEL_PORT"])

    if _is_server_running(host, port):
        logger.info(f"LlamaCpp running at {host}:{port}")
        return

    logger.info("Starting LlamaCpp.")

    log_file_path = os.path.expanduser(os.environ["LLAMACPP_MAIN_PATH_LOG_FOLDER"])
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    f_log = open(log_file_path, "w")

    # TODO: put this in cfg
    subprocess.Popen(
        [
            "./run_llamacpp/main.sh",
            "mode=server_plus_embed",
            "model=llama31instr_q5",
            "embed_model=nomic_embed",
        ],
        stdout=f_log,
        stderr=f_log,
        cwd=os.path.expanduser(os.environ["LLAMACPP_MAIN_PATH_RUN_DIR"]),
    )

    for _ in range(60):
        if _is_server_running(host, port):
            logger.info("LlamaCpp started.")
            return
        time.sleep(1)

    raise RuntimeError("LlamaCpp failed to start.")


def start_services(cfg: dict):
    log_fld = Path(os.environ["PROJECT_ROOT"]) / "logs"
    log_fld.mkdir(parents=True, exist_ok=True)

    f_pg = open(log_fld / "pgvector.log", "w")
    f_redis = open(log_fld / "redis.log", "w")
    f_qdrant = open(log_fld / "qdrant.log", "w")
    f_lf = open(log_fld / "langfuse.log", "w")
    f_ml = open(log_fld / "mlflow.log", "w")

    logs = [f_pg, f_redis, f_qdrant, f_lf, f_ml]
    langfuse_wd_path = None

    srv = cfg.get("services", {})

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {}

        if srv.get("langfuse", {}).get("start"):
            futures[executor.submit(_ensure_langfuse, f_lf)] = "Langfuse"

        if srv.get("llamacpp", {}).get("start"):
            futures[executor.submit(_ensure_llamacpp)] = "LlamaCpp"

        if srv.get("mlflow", {}).get("start"):
            futures[executor.submit(_ensure_mlflow, cfg, f_ml)] = "MLflow"

        if srv.get("qdrant", {}).get("start"):
            futures[executor.submit(_ensure_qdrant, f_qdrant)] = "Qdrant"

        if srv.get("pgvector", {}).get("start"):
            futures[executor.submit(_ensure_pgvector, f_pg)] = "PGVector"

        if srv.get("redis", {}).get("start"):
            futures[executor.submit(_ensure_redis, f_redis)] = "Redis"

        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
                if name == "Langfuse":
                    langfuse_wd_path = result
            except Exception as e:
                logger.error(f"Service {name} failed to start: {e}")

    return {
        "logs": logs,
        "langfuse_wd_path": langfuse_wd_path,
    }

def stop_services(
    state: dict,
    *,
    stop_all: bool = False,
    stop_langfuse: bool = False,
    stop_mlflow: bool = False,
    stop_llama_server: bool = False,
    stop_pgvector: bool = False,
    stop_redis: bool = False,
    stop_qdrant: bool = False,
):
    redirect = subprocess.DEVNULL

    def safe_run(cmd, *, cwd=None, name="service"):
        try:
            subprocess.run(cmd, stdout=redirect, stderr=redirect, cwd=cwd)
            logger.info(f"Stopped {name}.")
        except Exception as e:
            logger.error(f"Failed to stop {name}: {e}")

    if stop_llama_server or stop_all:
        safe_run(["pkill", "-f", "llama-server"], name="llama-server")

    if stop_mlflow or stop_all:
        safe_run(["pkill", "-f", "mlflow"], name="mlflow")

    if stop_pgvector or stop_all:
        safe_run(["docker", "compose", "stop", "pgvector"], name="PGVector")

    if stop_redis or stop_all:
        safe_run(["docker", "compose", "stop", "redis_cache"], name="Redis")

    if stop_qdrant or stop_all:
        safe_run(["docker", "compose", "stop", "qdrant"], name="Qdrant")

    if stop_langfuse or stop_all:
        if state.get("langfuse_wd_path"):
            safe_run(
                ["docker", "compose", "-f", "docker-compose.yml", "down"],
                cwd=state["langfuse_wd_path"],
                name="Langfuse",
            )

    for f in state.get("logs", []):
        f.close()


@contextmanager
def services_handler(cfg: dict, **stop_flags):
    state = start_services(cfg)
    try:
        yield state
    finally:
        stop_services(state, **stop_flags)