"""Centralized datetime conversion utilities for consistent ISO format handling across the application."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Union
from bson.objectid import ObjectId

logger = logging.getLogger(__name__)


def to_isoformat(obj: Any) -> Any:
    """Convert datetime objects and nested structures to ISO format strings.
    
    Also converts MongoDB ObjectId to string representation.
    
    Recursively handles:
    - Direct datetime objects → ISO string
    - ObjectId objects → string representation
    - Dictionaries with datetime/ObjectId values
    - Lists containing datetime/ObjectId objects
    - Mixed nested structures
    
    Args:
        obj: Any Python object (datetime, ObjectId, dict, list, or primitive)
    
    Returns:
        Object with all datetime instances converted to ISO format strings
        and all ObjectId instances converted to strings
        
    Example:
        >>> result = to_isoformat({
        ...     'timestamp': datetime.now(),
        ...     '_id': ObjectId(),
        ...     'items': [datetime.now(), 'text']
        ... })
        >>> assert isinstance(result['timestamp'], str)
        >>> assert isinstance(result['_id'], str)
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, dict):
        return {key: to_isoformat(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [to_isoformat(item) for item in obj]
    else:
        return obj


def from_isoformat(obj: Any) -> Any:
    """Convert ISO format strings back to datetime objects.
    
    Recursively handles:
    - ISO format strings → datetime objects
    - Dictionaries with ISO string values
    - Lists containing ISO strings
    - Mixed nested structures
    
    Args:
        obj: Any Python object (string, dict, list, or primitive)
    
    Returns:
        Object with all ISO format strings converted to datetime instances
        
    Raises:
        ValueError: If ISO format string is invalid
        
    Example:
        >>> iso_str = datetime.now().isoformat()
        >>> dt = from_isoformat(iso_str)
        >>> assert isinstance(dt, datetime)
    """
    if isinstance(obj, str):
        # Heuristic: ISO format strings are typically > 18 chars and contain 'T'
        if len(obj) > 18 and 'T' in obj:
            try:
                return datetime.fromisoformat(obj)
            except ValueError:
                # Not a datetime string, return as-is
                return obj
        return obj
    elif isinstance(obj, dict):
        return {key: from_isoformat(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [from_isoformat(item) for item in obj]
    else:
        return obj


def ensure_datetime_fields(obj: Dict, fields: List[str], as_iso: bool = False) -> Dict:
    """Ensure specific fields in a dictionary are datetime objects (or ISO strings).
    
    Useful for converting datetime fields to/from ISO format before database operations.
    
    Args:
        obj: Dictionary to process
        fields: List of field names to convert
        as_iso: If True, convert to ISO strings; if False, convert to datetime objects
    
    Returns:
        Dictionary with specified fields converted
        
    Example:
        >>> profile = {
        ...     'created_at': '2025-12-01T10:00:00',
        ...     'name': 'John'
        ... }
        >>> ensure_datetime_fields(profile, ['created_at'], as_iso=False)
        >>> assert isinstance(profile['created_at'], datetime)
    """
    result = obj.copy()
    converter = to_isoformat if as_iso else from_isoformat
    
    for field in fields:
        if field in result:
            result[field] = converter(result[field])
    
    return result


# Backward compatibility alias (for existing code that calls the old function name)
convert_datetime_to_isoformat = to_isoformat

__all__ = ['to_isoformat', 'from_isoformat', 'ensure_datetime_fields', 'convert_datetime_to_isoformat']
