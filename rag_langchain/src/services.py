import socket
import os
import sys
import time
import logging
import subprocess
import mlflow
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed

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


@contextmanager
def services_handler(
    cfg: dict,
    stop_all: bool = False,
    stop_langfuse: bool = False,
    stop_mlflow: bool = False,
    stop_llama_server: bool = False,
    stop_pgvector: bool = False,
    stop_redis: bool = False,
    stop_qdrant: bool = False,
):
    os.makedirs("logs", exist_ok=True)
    f_pg = open("logs/pgvector.log", "w")
    f_redis = open("logs/redis.log", "w")
    f_qdrant = open("logs/qdrant.log", "w")
    f_lf = open("logs/langfuse.log", "w")
    f_ml = open("logs/mlflow.log", "w")
    logs = [f_pg, f_redis, f_qdrant, f_lf, f_ml]
    langfuse_wd_path = None

    # Check what needs to be started based on config
    srv = cfg.get("services", {})
    start_pg = srv.get("pgvector", {}).get("start", False)
    start_redis = srv.get("redis", {}).get("start", False)
    start_qdrant = srv.get("qdrant", {}).get("start", False)
    start_llama = srv.get("llamacpp", {}).get("start", False)
    start_mlflow = srv.get("mlflow", {}).get("start", False)
    start_langfuse = srv.get("langfuse", {}).get("start", False)

    try:
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {}

            if start_langfuse:
                futures[executor.submit(_ensure_langfuse, f_lf)] = "Langfuse"

            if start_llama:
                futures[executor.submit(_ensure_llamacpp)] = "LlamaCpp"

            if start_mlflow:
                futures[executor.submit(_ensure_mlflow, cfg, f_ml)] = "MLflow"

            if start_qdrant:
                futures[executor.submit(_ensure_qdrant, f_qdrant)] = "Qdrant"

            if start_pg:
                futures[executor.submit(_ensure_pgvector, f_pg)] = "PGVector"

            if start_redis:
                futures[executor.submit(_ensure_redis, f_redis)] = "Redis"

            for future in as_completed(futures):
                service_name = futures[future]
                try:
                    result = future.result()
                    if service_name == "Langfuse":
                        langfuse_wd_path = result
                except Exception as e:
                    logger.error(f"Service {service_name} failed to start: {e}")

        yield

    finally:
        redirect = subprocess.DEVNULL

        if stop_llama_server or stop_all:
            try:
                subprocess.run("pkill -f llama-server", shell=True, stdout=redirect)
                logger.info("Stopped llama-server.")
            except Exception as e:
                logger.error(f"Failed to stop llama-server: {e}")

        if stop_mlflow or stop_all:
            try:
                subprocess.run("pkill -f mlflow", shell=True, stdout=redirect)
                logger.info("Stopped py-spawned services.")
            except Exception as e:
                logger.error(f"Failed to stop mlflow: {e}")

        if stop_pgvector or stop_all:
            try:
                subprocess.run(
                    ["docker", "compose", "stop", "pgvector"],
                    stdout=redirect,
                    stderr=redirect,
                )
                logger.info("Stopped PGVector.")
            except Exception as e:
                logger.error(f"Failed to stop PGVector: {e}")

        if stop_redis or stop_all:
            try:
                subprocess.run(
                    ["docker", "compose", "stop", "redis_cache"],
                    stdout=redirect,
                    stderr=redirect,
                )
                logger.info("Stopped Redis.")
            except Exception as e:
                logger.error(f"Failed to stop Redis: {e}")

        if stop_qdrant or stop_all:
            try:
                subprocess.run(
                    ["docker", "compose", "stop", "qdrant"],
                    stdout=redirect,
                    stderr=redirect,
                )
                logger.info("Stopped Qdrant.")
            except Exception as e:
                logger.error(f"Failed to stop Qdrant: {e}")

        if stop_langfuse or stop_all:
            try:
                if langfuse_wd_path:
                    subprocess.run(
                        ["docker", "compose", "-f", "docker-compose.yml", "down"],
                        stdout=redirect,
                        stderr=redirect,
                        cwd=langfuse_wd_path,
                    )
                    logger.info("Stopped Langfuse.")
            except Exception as e:
                logger.error(f"Failed to stop Langfuse: {e}")

        _ = [ff.close() for ff in logs]