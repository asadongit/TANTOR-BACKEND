from typing import List, Optional, Union
from pydantic import BaseModel, Field


class RiskBand(BaseModel):
    """Defines a score range mapped to a risk label and letter grade"""

    min_score: float = Field(
        ..., description="Minimum normalized score (0-100) for this band"
    )
    max_score: float = Field(
        ..., description="Maximum normalized score (0-100) for this band"
    )
    label: str = Field(..., description="Risk label (e.g., 'Low Risk', 'High Risk')")
    grade: str = Field(..., description="Letter grade (e.g., 'A', 'B', 'F')")


class CrossCategoryRule(BaseModel):
    """Rule evaluated across category scores"""

    condition: str = Field(
        ...,
        description="SimpleEval expression referencing categories, e.g., 'category_a.normalized_score > 50'",
    )
    penalty_weight: float = Field(
        ..., description="Score adjustment (penalty or bonus) applied to total score"
    )
    rule_id: Optional[str] = Field(None, description="Identifier for explainability")
    enabled: bool = Field(True, description="Whether this rule is active")


class WeightageCondition(BaseModel):
    """A condition and its corresponding weight score"""

    condition: str = Field(
        ..., description="Condition expression to evaluate, e.g., 'loan_amount > 5000'"
    )
    weight: Union[float, str] = Field(
        ...,
        description="Weight score to add if condition evaluates to True, can be a float or dynamic string",
    )
    enabled: bool = Field(
        True, description="Whether this condition is active and evaluated"
    )
    rule_id: Optional[str] = Field(
        None, description="Standardized ID/Code for this rule (explainability)"
    )


class WeightageField(BaseModel):
    """A field containing multiple conditions"""

    name: str = Field(..., description="Field name or JSON path (e.g. 'loan_amount')")
    match_strategy: str = Field(
        "first_match",
        description="Strategy: first_match (stop on first matched condition) or sum_all (sum all matched conditions)",
    )
    default_weight: float = Field(
        0.0, description="Fallback weight if no conditions match"
    )
    conditions: List[WeightageCondition] = Field(
        default_factory=list, description="Conditions for this field"
    )
    cap_score: Optional[float] = Field(
        None, description="Optional maximum score this field can contribute"
    )
    enabled: bool = Field(
        True, description="Whether this field is active and evaluated"
    )


class WeightageCategory(BaseModel):
    """A category containing multiple fields"""

    name: str = Field(..., description="Category name (e.g. 'loan')")
    fields: List[WeightageField] = Field(
        default_factory=list, description="Fields in this category"
    )
    weight: float = Field(1.0, description="Category-level weight multiplier")
    is_kill_switch: bool = Field(
        False,
        description="If True, any non-zero score in this category will trigger a reject decision",
    )
    kill_switch_score: float = Field(
        100.0, description="The score penalty or rating associated with this switch"
    )
    fail_threshold: Optional[float] = Field(
        None, description="Minimum normalized score required for this category to pass"
    )
    enabled: bool = Field(True, description="Whether this category is active")

