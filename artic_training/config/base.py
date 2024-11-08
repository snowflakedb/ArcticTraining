from pydantic import BaseModel
from pydantic import ConfigDict


class BaseConfig(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        use_enum_values=True,
        validate_assignment=True,
        validate_default=True,
        use_attribute_docstrings=True,
    )
