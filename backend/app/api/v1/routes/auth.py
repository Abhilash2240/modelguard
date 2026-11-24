from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db
from app.core.config import settings
from app.core.security import create_access_token, verify_password
from app.db import models
from app.schemas.auth import LoginRequest, TokenResponse, UserProfile

router = APIRouter(prefix="/auth", tags=["Authentication & Access Control"])


@router.post("/login", response_model=TokenResponse)
def login(
    payload: LoginRequest,
    response: Response,
    db: Session = Depends(get_db),
) -> TokenResponse:
    """Authenticate with username and password, return signed JWT, and set HttpOnly session cookie."""
    user = db.query(models.User).filter(models.User.username == payload.username).first()
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive or disabled.",
        )

    token = create_access_token(subject=user.username, role=user.role)

    # Set secure HttpOnly cookie for browser sessions
    response.set_cookie(
        key="mg_access_token",
        value=token,
        httponly=True,
        samesite="lax",
        secure=(settings.ENVIRONMENT == "production"),
        max_age=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        user=UserProfile.model_validate(user),
    )


@router.post("/logout")
def logout(response: Response) -> dict[str, str]:
    """Clear session cookie and invalidate active browser session."""
    response.delete_cookie(key="mg_access_token")
    return {"message": "Successfully logged out."}


@router.get("/me", response_model=UserProfile)
def get_current_user_profile(
    current_user: models.User = Depends(get_current_user),
) -> UserProfile:
    """Return identity and assigned role of currently authenticated principal."""
    return UserProfile.model_validate(current_user)
