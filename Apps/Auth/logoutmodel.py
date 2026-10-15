# app/schemas/connection.py
from pydantic import BaseModel, Field

class Logout(BaseModel):
    refresh_token: str = Field(repr=False)
