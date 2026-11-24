from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=100, description="User login identifier")
    password: str = Field(..., min_length=1, description="Plaintext password to authenticate")


class UserProfile(BaseModel):
    id: str
    username: str
    role: str
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserProfile


class CreateUserRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=100)
    password: str = Field(..., min_length=8)
    role: Literal["VIEWER", "OPERATOR", "APPROVER", "ADMIN"] = "VIEWER"
