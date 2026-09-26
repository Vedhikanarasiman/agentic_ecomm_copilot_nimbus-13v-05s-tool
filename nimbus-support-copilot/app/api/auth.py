from fastapi import APIRouter
from pydantic import BaseModel

from app.core.security import create_access_token

router = APIRouter(prefix="/auth", tags=["auth"])


class TokenRequest(BaseModel):
    customer_id: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


@router.post("/token", response_model=TokenResponse)
async def issue_token(req: TokenRequest):
    # NOTE: no password/identity check here. A real system verifies the
    # customer before issuing a token; this endpoint exists to demonstrate
    # the JWT-protected request path, not to be a real auth system.
    token = create_access_token(customer_id=req.customer_id)
    return TokenResponse(access_token=token)
