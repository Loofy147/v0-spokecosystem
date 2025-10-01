"""
Schema validation for complex data structures.
"""
from typing import Any, Dict, List, Optional, Union
from utils.logging_config import get_logger
from utils.exceptions import ValidationError

logger = get_logger(__name__)


class Schema:
    """Schema for validating dictionaries."""
    
    def __init__(self, schema: Dict[str, Any], strict: bool = False):
        """
        Initialize schema.
        
        Args:
            schema: Dictionary defining expected structure
            strict: If True, reject extra keys not in schema
        """
        self.schema = schema
        self.strict = strict
    
    def validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data against schema."""
        if not isinstance(data, dict):
            raise ValidationError(
                f"Expected dict, got {type(data)}",
                details={'actual_type': str(type(data))}
            )
        
        validated = {}
        errors = []
        
        # Check required fields
        for key, spec in self.schema.items():
            if key not in data:
                if isinstance(spec, dict) and not spec.get('required', True):
                    # Optional field, use default if provided
                    if 'default' in spec:
                        validated[key] = spec['default']
                    continue
                else:
                    errors.append(f"Missing required field: {key}")
                    continue
            
            # Validate field
            try:
                validated[key] = self._validate_field(key, data[key], spec)
            except ValidationError as e:
                errors.append(f"Field '{key}': {e.message}")
        
        # Check for extra fields in strict mode
        if self.strict:
            extra_keys = set(data.keys()) - set(self.schema.keys())
            if extra_keys:
                errors.append(f"Unexpected fields: {extra_keys}")
        else:
            # Include extra fields
            for key in data:
                if key not in validated:
                    validated[key] = data[key]
        
        if errors:
            raise ValidationError(
                "Schema validation failed",
                details={'errors': errors}
            )
        
        return validated
    
    def _validate_field(self, key: str, value: Any, spec: Any) -> Any:
        """Validate a single field."""
        # If spec is a type, check type
        if isinstance(spec, type):
            if not isinstance(value, spec):
                raise ValidationError(
                    f"Expected {spec}, got {type(value)}",
                    details={'expected': str(spec), 'actual': str(type(value))}
                )
            return value
        
        # If spec is a dict with validation rules
        if isinstance(spec, dict):
            expected_type = spec.get('type')
            if expected_type and not isinstance(value, expected_type):
                raise ValidationError(
                    f"Expected {expected_type}, got {type(value)}",
                    details={'expected': str(expected_type), 'actual': str(type(value))}
                )
            
            # Apply validators
            if 'validator' in spec:
                validator = spec['validator']
                value = validator.validate(value)
            
            # Check choices
            if 'choices' in spec and value not in spec['choices']:
                raise ValidationError(
                    f"Must be one of {spec['choices']}, got {value}",
                    details={'allowed': spec['choices'], 'actual': value}
                )
            
            return value
        
        return value


class ConfigSchema(Schema):
    """Schema specifically for configuration validation."""
    
    def __init__(self):
        schema = {
            'learning_rate': {
                'type': float,
                'required': True,
                'validator': RangeValidator(min_value=0, inclusive=False, name='learning_rate')
            },
            'batch_size': {
                'type': int,
                'required': True,
                'validator': RangeValidator(min_value=1, name='batch_size')
            },
            'epochs': {
                'type': int,
                'required': True,
                'validator': RangeValidator(min_value=1, name='epochs')
            },
            'optimizer': {
                'type': str,
                'required': False,
                'default': 'adam',
                'choices': ['adam', 'sgd', 'rmsprop', 'adamw']
            },
            'device': {
                'type': str,
                'required': False,
                'default': 'cpu',
                'choices': ['cpu', 'cuda', 'mps']
            }
        }
        super().__init__(schema, strict=False)


# Import validators for use in schema
from .validators import RangeValidator
