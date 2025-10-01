"""
Error handling utilities and decorators.
"""
import functools
import time
from typing import Any, Callable, Optional, Type, Tuple
from .logging_config import get_logger
from .exceptions import SpokeEcosystemError, TimeoutError as SpokeTimeoutError


logger = get_logger(__name__)


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
):
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each retry
        exceptions: Tuple of exception types to catch and retry
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {e}. "
                            f"Retrying in {current_delay}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}",
                            exc_info=True
                        )
            
            raise last_exception
        
        return wrapper
    return decorator


def timeout(seconds: float):
    """
    Decorator for adding timeout to functions.
    
    Args:
        seconds: Timeout in seconds
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import signal
            
            def timeout_handler(signum, frame):
                raise SpokeTimeoutError(
                    f"Function {func.__name__} timed out after {seconds}s"
                )
            
            # Set the signal handler and alarm
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(seconds))
            
            try:
                result = func(*args, **kwargs)
            finally:
                # Restore the old handler and cancel the alarm
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            
            return result
        
        return wrapper
    return decorator


def safe_execute(
    func: Callable,
    *args,
    default: Any = None,
    log_errors: bool = True,
    **kwargs
) -> Any:
    """
    Safely execute a function and return default value on error.
    
    Args:
        func: Function to execute
        *args: Positional arguments for function
        default: Default value to return on error
        log_errors: Whether to log errors
        **kwargs: Keyword arguments for function
    
    Returns:
        Function result or default value
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            logger.error(f"Error executing {func.__name__}: {e}", exc_info=True)
        return default


def handle_errors(
    default_return: Any = None,
    raise_on_error: bool = False,
    log_level: str = "ERROR"
):
    """
    Decorator for handling errors in functions.
    
    Args:
        default_return: Value to return on error
        raise_on_error: Whether to re-raise the exception
        log_level: Logging level for errors
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except SpokeEcosystemError as e:
                # Log structured error
                log_method = getattr(logger, log_level.lower())
                log_method(
                    f"SpokeEcosystem error in {func.__name__}: {e.message}",
                    extra={'error_details': e.to_dict()}
                )
                if raise_on_error:
                    raise
                return default_return
            except Exception as e:
                # Log unexpected error
                log_method = getattr(logger, log_level.lower())
                log_method(
                    f"Unexpected error in {func.__name__}: {e}",
                    exc_info=True
                )
                if raise_on_error:
                    raise
                return default_return
        
        return wrapper
    return decorator


class ErrorContext:
    """Context manager for error handling with cleanup."""
    
    def __init__(
        self,
        operation_name: str,
        cleanup_func: Optional[Callable] = None,
        raise_on_error: bool = True
    ):
        self.operation_name = operation_name
        self.cleanup_func = cleanup_func
        self.raise_on_error = raise_on_error
        self.logger = get_logger(__name__)
    
    def __enter__(self):
        self.logger.info(f"Starting operation: {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.logger.error(
                f"Error in operation {self.operation_name}: {exc_val}",
                exc_info=(exc_type, exc_val, exc_tb)
            )
            
            # Run cleanup if provided
            if self.cleanup_func:
                try:
                    self.cleanup_func()
                except Exception as cleanup_error:
                    self.logger.error(
                        f"Error during cleanup: {cleanup_error}",
                        exc_info=True
                    )
            
            # Suppress exception if raise_on_error is False
            return not self.raise_on_error
        else:
            self.logger.info(f"Completed operation: {self.operation_name}")
            return False
