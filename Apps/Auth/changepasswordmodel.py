from pydantic import BaseModel, Field

class ChangePassword(BaseModel):
    username: str
    old_password: str = Field(repr=False)
    new_password: str = Field(repr=False)
