"""
Comprehensive input validation utilities.
"""
from typing import Any, Callable, List, Optional, Tuple, Union
import numpy as np
from utils.logging_config import get_logger
from utils.exceptions import ValidationError

logger = get_logger(__name__)


class Validator:
    """Base validator class."""
    
    def __init__(self, name: str = "value"):
        self.name = name
    
    def validate(self, value: Any) -> Any:
        """Validate and return the value."""
        return value
    
    def __call__(self, value: Any) -> Any:
        """Allow validator to be called as a function."""
        return self.validate(value)


class TypeValidator(Validator):
    """Validates value type."""
    
    def __init__(self, expected_type: Union[type, Tuple[type, ...]], name: str = "value"):
        super().__init__(name)
        self.expected_type = expected_type
    
    def validate(self, value: Any) -> Any:
        """Validate type."""
        if not isinstance(value, self.expected_type):
            raise ValidationError(
                f"{self.name} must be of type {self.expected_type}, got {type(value)}",
                details={'expected': str(self.expected_type), 'actual': str(type(value))}
            )
        return value


class RangeValidator(Validator):
    """Validates numeric value is within range."""
    
    def __init__(
        self,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        inclusive: bool = True,
        name: str = "value"
    ):
        super().__init__(name)
        self.min_value = min_value
        self.max_value = max_value
        self.inclusive = inclusive
    
    def validate(self, value: Union[int, float]) -> Union[int, float]:
        """Validate range."""
        if self.min_value is not None:
            if self.inclusive and value < self.min_value:
                raise ValidationError(
                    f"{self.name} must be >= {self.min_value}, got {value}"
                )
            elif not self.inclusive and value <= self.min_value:
                raise ValidationError(
                    f"{self.name} must be > {self.min_value}, got {value}"
                )
        
        if self.max_value is not None:
            if self.inclusive and value > self.max_value:
                raise ValidationError(
                    f"{self.name} must be <= {self.max_value}, got {value}"
                )
            elif not self.inclusive and value >= self.max_value:
                raise ValidationError(
                    f"{self.name} must be < {self.max_value}, got {value}"
                )
        
        return value


class ShapeValidator(Validator):
    """Validates array/tensor shape."""
    
    def __init__(
        self,
        expected_shape: Optional[Tuple[Optional[int], ...]] = None,
        ndim: Optional[int] = None,
        name: str = "array"
    ):
        super().__init__(name)
        self.expected_shape = expected_shape
        self.ndim = ndim
    
    def validate(self, value: np.ndarray) -> np.ndarray:
        """Validate shape."""
        if not isinstance(value, np.ndarray):
            raise ValidationError(
                f"{self.name} must be a numpy array, got {type(value)}"
            )
        
        # Check number of dimensions
        if self.ndim is not None and value.ndim != self.ndim:
            raise ValidationError(
                f"{self.name} must have {self.ndim} dimensions, got {value.ndim}",
                details={'expected_ndim': self.ndim, 'actual_shape': value.shape}
            )
        
        # Check specific shape
        if self.expected_shape is not None:
            if len(value.shape) != len(self.expected_shape):
                raise ValidationError(
                    f"{self.name} shape mismatch: expected {len(self.expected_shape)} dimensions, "
                    f"got {len(value.shape)}",
                    details={'expected': self.expected_shape, 'actual': value.shape}
                )
            
            for i, (expected, actual) in enumerate(zip(self.expected_shape, value.shape)):
                if expected is not None and expected != actual:
                    raise ValidationError(
                        f"{self.name} shape mismatch at dimension {i}: "
                        f"expected {expected}, got {actual}",
                        details={'expected': self.expected_shape, 'actual': value.shape}
                    )
        
        return value


class ChoiceValidator(Validator):
    """Validates value is in allowed choices."""
    
    def __init__(self, choices: List[Any], name: str = "value"):
        super().__init__(name)
        self.choices = choices
    
    def validate(self, value: Any) -> Any:
        """Validate choice."""
        if value not in self.choices:
            raise ValidationError(
                f"{self.name} must be one of {self.choices}, got {value}",
                details={'allowed': self.choices, 'actual': value}
            )
        return value


class LengthValidator(Validator):
    """Validates length of sequence."""
    
    def __init__(
        self,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        exact_length: Optional[int] = None,
        name: str = "sequence"
    ):
        super().__init__(name)
        self.min_length = min_length
        self.max_length = max_length
        self.exact_length = exact_length
    
    def validate(self, value: Any) -> Any:
        """Validate length."""
        length = len(value)
        
        if self.exact_length is not None and length != self.exact_length:
            raise ValidationError(
                f"{self.name} must have exactly {self.exact_length} elements, got {length}"
            )
        
        if self.min_length is not None and length < self.min_length:
            raise ValidationError(
                f"{self.name} must have at least {self.min_length} elements, got {length}"
            )
        
        if self.max_length is not None and length > self.max_length:
            raise ValidationError(
                f"{self.name} must have at most {self.max_length} elements, got {length}"
            )
        
        return value


class CompositeValidator(Validator):
    """Combines multiple validators."""
    
    def __init__(self, *validators: Validator, name: str = "value"):
        super().__init__(name)
        self.validators = validators
    
    def validate(self, value: Any) -> Any:
        """Apply all validators in sequence."""
        for validator in self.validators:
            value = validator.validate(value)
        return value


class CustomValidator(Validator):
    """Validator using custom function."""
    
    def __init__(self, func: Callable[[Any], bool], error_message: str, name: str = "value"):
        super().__init__(name)
        self.func = func
        self.error_message = error_message
    
    def validate(self, value: Any) -> Any:
        """Validate using custom function."""
        if not self.func(value):
            raise ValidationError(f"{self.name}: {self.error_message}")
        return value


def validate_positive(value: Union[int, float], name: str = "value") -> Union[int, float]:
    """Validate value is positive."""
    return RangeValidator(min_value=0, inclusive=False, name=name).validate(value)


def validate_non_negative(value: Union[int, float], name: str = "value") -> Union[int, float]:
    """Validate value is non-negative."""
    return RangeValidator(min_value=0, inclusive=True, name=name).validate(value)


def validate_probability(value: float, name: str = "probability") -> float:
    """Validate value is a valid probability [0, 1]."""
    return RangeValidator(min_value=0.0, max_value=1.0, inclusive=True, name=name).validate(value)


def validate_shape_compatible(
    arr1: np.ndarray,
    arr2: np.ndarray,
    operation: str = "operation"
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate two arrays have compatible shapes."""
    if arr1.shape != arr2.shape:
        raise ValidationError(
            f"Shape mismatch for {operation}: {arr1.shape} vs {arr2.shape}",
            details={'shape1': arr1.shape, 'shape2': arr2.shape}
        )
    return arr1, arr2
