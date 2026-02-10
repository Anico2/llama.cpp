from functools import wraps
import time

def log_execution(logger):
    """Decorator factory to log function name and execution time."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            strategy_name = args[0].__class__.__name__
            logger.info(f"Starting {strategy_name} for query...")
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logger.info(f"{strategy_name} finished in {duration:.2f}s")
            return result

        return wrapper

    return decorator
