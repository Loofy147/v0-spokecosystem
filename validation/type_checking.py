"""
Runtime type checking utilities.
"""
from typing import Any, Callable, get_type_hints, Union
import functools
import inspect
from utils.logging_config import get_logger
from utils.exceptions import ValidationError

logger = get_logger(__name__)


def validate_types(func: Callable) -> Callable:
    """
    Decorator for runtime type checking based on type hints.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Get type hints
        hints = get_type_hints(func)
        
        # Get function signature
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        
        # Validate each argument
        for param_name, param_value in bound_args.arguments.items():
            if param_name in hints:
                expected_type = hints[param_name]
                
                # Skip if type is Any
                if expected_type is Any:
                    continue
                
                # Handle Union types
                if hasattr(expected_type, '__origin__') and expected_type.__origin__ is Union:
                    valid_types = expected_type.__args__
                    if not isinstance(param_value, valid_types):
                        raise ValidationError(
                            f"Parameter '{param_name}' must be one of {valid_types}, "
                            f"got {type(param_value)}",
                            details={
                                'parameter': param_name,
                                'expected': str(valid_types),
                                'actual': str(type(param_value))
                            }
                        )
                else:
                    # Simple type check
                    if not isinstance(param_value, expected_type):
                        raise ValidationError(
                            f"Parameter '{param_name}' must be {expected_type}, "
                            f"got {type(param_value)}",
                            details={
                                'parameter': param_name,
                                'expected': str(expected_type),
                                'actual': str(type(param_value))
                            }
                        )
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Validate return type if specified
        if 'return' in hints:
            expected_return = hints['return']
            if expected_return is not Any and not isinstance(result, expected_return):
                logger.warning(
                    f"Return type mismatch in {func.__name__}: "
                    f"expected {expected_return}, got {type(result)}"
                )
        
        return result
    
    return wrapper


def ensure_type(value: Any, expected_type: type, name: str = "value") -> Any:
    """
    Ensure value is of expected type, raise ValidationError if not.
    """
    if not isinstance(value, expected_type):
        raise ValidationError(
            f"{name} must be {expected_type}, got {type(value)}",
            details={'expected': str(expected_type), 'actual': str(type(value))}
        )
    return value


def coerce_type(value: Any, target_type: type, name: str = "value") -> Any:
    """
    Attempt to coerce value to target type.
    """
    try:
        return target_type(value)
    except (ValueError, TypeError) as e:
        raise ValidationError(
            f"Cannot convert {name} to {target_type}: {e}",
            details={'value': str(value), 'target_type': str(target_type)}
        )
