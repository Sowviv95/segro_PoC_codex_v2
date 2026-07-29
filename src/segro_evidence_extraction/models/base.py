"""Shared model defaults for evidence-first contracts."""

from pydantic import BaseModel, ConfigDict


class StrictBaseModel(BaseModel):
    """Base model with strict extra-field handling and string normalization."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        use_enum_values=True,
    )
