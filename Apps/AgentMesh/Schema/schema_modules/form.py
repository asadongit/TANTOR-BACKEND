from typing import Any, List, Optional
from pydantic import BaseModel, Field


class FormField(BaseModel):
    """Configuration for a single field in a Form Submission node"""

    id: str = Field(
        ..., description="Unique ID for the field (used as key in context/state)"
    )
    label: str = Field(..., description="Display label for the form field")
    type: str = Field(
        "text",
        description="Field type: text, email, password, number, textarea, select, multiselect, radio, date, boolean, file, remark",
    )
    placeholder: Optional[str] = None
    default_value: Any = None
    content: Optional[str] = Field(
        None, description="Static text content for 'remark' type fields"
    )
    required: bool = True
    enabled: bool = Field(True, description="Whether this field is active")
    options: Optional[List[str]] = Field(
        None, description="For select/multiselect/radio: ['Option1', 'Option2']"
    )
    validation_regex: Optional[str] = None

    # File upload specific configs
    allowed_types: Optional[List[str]] = Field(
        None,
        description="Allowed file types: pdf, csv, excel, word, image, video, zip, json. None = all types.",
    )
    file_storage_mode: Optional[str] = Field(
        "path",
        description="File storage mode: 'path' (save to temp dir) or 'binary' (inline Base64)",
    )
    max_size_mb: Optional[int] = Field(10, description="Maximum file size in MB")
    allow_multiple: Optional[bool] = Field(
        False, description="Allow multiple file uploads"
    )


class FormDecision(BaseModel):
    """Configuration for a custom decision button in a Form Submission node"""

    id: str = Field(
        ..., description="Unique ID / value for the decision (stored in context/state)"
    )
    label: str = Field(..., description="Display label for the decision button")
