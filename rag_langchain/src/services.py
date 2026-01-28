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


def _ensure_pgvector(log_file):
    host = os.environ["PG_HOST_"]
    port = int(os.environ["PG_PORT_"])

    if _is_server_running(host, port):
        logger.info(f"PGVector running at {host}:{port}")
        return

    logger.info("Starting PGVector.")

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


def _ensure_langfuse(log_file):
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


def _ensure_mlflow(cfg, log_file):
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


def _ensure_llamacpp():
    # LlamaCpp handles its own logging logic via env var path
    host = os.environ["MODEL_HOST"]
    port = int(os.environ["MODEL_PORT"])

    if _is_server_running(host, port):
        logger.info(f"LlamaCpp running at {host}:{port}")
        return

    logger.info("Starting LlamaCpp.")

    log_file_path = os.path.expanduser(os.environ["LLAMACPP_MAIN_PATH_LOG_FOLDER"])
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    f_log = open(log_file_path, "w")

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
    cfg,
    suppress_out=True,  # Kept arg for compatibility, but ignored in favor of logging
    stop_all=False,
    stop_langfuse=False,
    stop_mlflow=False,
    stop_llama_server=False,
    stop_pgvector=False,
):
    os.makedirs("logs", exist_ok=True)
    f_pg = open("logs/pgvector.log", "w")
    f_lf = open("logs/langfuse.log", "w")
    f_ml = open("logs/mlflow.log", "w")
    

    langfuse_wd_path = None

    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_llama = executor.submit(_ensure_llamacpp)
            future_pg = executor.submit(_ensure_pgvector, f_pg)
            future_mlflow = executor.submit(_ensure_mlflow, cfg, f_ml)
            future_langfuse = executor.submit(_ensure_langfuse, f_lf)

            futures = {
                future_langfuse: "Langfuse",
                future_llama: "LlamaCpp",
                future_pg: "PGVector",
                future_mlflow: "MLflow",
            }

            for future in as_completed(futures):
                service_name = futures[future]
                try:
                    result = future.result()
                    if service_name == "Langfuse":
                        langfuse_wd_path = result

                except Exception as e:
                    logger.error(f"Service {service_name} failed to start: {e}")
                    raise RuntimeError(f"Critical service {service_name} failed setup.") from e
        
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
                logger.info("Stopped docker-spawned services (PGVector).")
                time.sleep(10)
            except Exception as e:
                logger.error(f"Failed to stop PGVector: {e}")

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
                else:
                    logger.warning("Skipping Langfuse shutdown: Path not resolved.")
            except Exception as e:
                logger.error(f"Failed to stop Langfuse: {e}")
        
        
        _ = f_pg.close(), f_lf.close(), f_ml.close()