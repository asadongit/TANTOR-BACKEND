from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import os
from keycloak import KeycloakOpenID
from pydantic_settings import BaseSettings
from .loginmodel import Login
from .logoutmodel import Logout
from dotenv import load_dotenv
import jwt
import logging

# Setup logger for this module
logger = logging.getLogger(__name__)

load_dotenv(".env")

router = APIRouter()


class KeycloakConfig(BaseSettings):
    server_url: str = os.getenv('KEYCLOAK_SERVER_URL')
    client_id: str = os.getenv('KEYCLOAK_CLIENT_ID')
    realm_name: str = os.getenv('KEYCLOAK_REALM_NAME')
    client_secret_key: str = os.getenv('KEYCLOAK_CLIENT_SECRET_KEY')


config = KeycloakConfig()

keycloak_openid = KeycloakOpenID(
    server_url=config.server_url,
    client_id=config.client_id,
    realm_name=config.realm_name,
    client_secret_key=config.client_secret_key,
    verify=False
)


@router.post("/login")
async def login(login: Login):
    try:
        token = keycloak_openid.token(login.username, login.password)

        decoded = jwt.decode(token['access_token'], options={
                             "verify_signature": False})
        print(decoded)
        return JSONResponse(content=token)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid credentials")


@router.post("/logout")
async def logout(logout: Logout):
    try:
        keycloak_openid.logout(logout.refresh_token)
        return JSONResponse(content={"message": "User logged out successfully."}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Logout failed: {str(e)}")


@router.post("/refresh")
async def refresh_access_token(request_body: dict):
    """
    Refresh expired access token using refresh token

     PURPOSE:
    When the access token expires (usually after 5-15 minutes),
    this endpoint gets a new access token without requiring login again.

    REQUEST BODY:
    {
        "refresh_token": "your_refresh_token_here"
    }

    RESPONSE:
    {
        "access_token": "new_access_token",
        "refresh_token": "new_or_same_refresh_token",
        "id_token": "new_id_token"
    }

     SECURITY:
    - No authentication required (uses refresh_token for auth)
    - Refresh token must be valid and not expired
    - Returns new tokens if successful
    - Returns 401 if refresh token is expired/invalid
    """
    try:
        # Extract refresh token from request body
        refresh_token = request_body.get("refresh_token")

        if not refresh_token:
            logger.warning("Refresh attempt without refresh_token")
            raise HTTPException(
                status_code=400, detail="Refresh token is required")

        logger.info("Attempting to refresh access token")

        new_tokens = keycloak_openid.refresh_token(refresh_token)

        logger.info("Access token refreshed successfully")

        return JSONResponse(content={
            "access_token": new_tokens.get("access_token"),
            "refresh_token": new_tokens.get("refresh_token", refresh_token),
            "id_token": new_tokens.get("id_token")
        }, status_code=200)

    except Exception as e:
        # Log the error
        logger.error(f"Token refresh failed: {str(e)}")

        raise HTTPException(
            status_code=401,
            detail="Refresh token expired or invalid. Please login again."
        )
