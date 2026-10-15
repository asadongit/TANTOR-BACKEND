import httpx
import logging
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from Database.Models import ConnectionProfileModel
from Apps.AgentMesh.Utilities.Security import (
    decrypt_string,
    encrypt_string,
    is_encrypted,
    validate_credential_destination,
)

logger = logging.getLogger(__name__)

class OAuthManager:
    """
    Enterprise Service responsible for retrieving and refreshing tokens securely.
    Implements the "Lazy Refresh" architecture to save CPU and DB resources.
    """
    
    @staticmethod
    async def get_valid_token(db: AsyncSession, connection_id: str) -> str:
        """
        Retrieves the access token from the vault. Automatically handles dead tokens.
        """
        # 1. Fetch connection from Vault
        stmt = select(ConnectionProfileModel).where(ConnectionProfileModel.id == connection_id)
        result = await db.execute(stmt)
        conn = result.scalar_one_or_none()
        
        if not conn:
            raise ValueError(f"Connection profile '{connection_id}' not found in registry.")
            
        # 2. Update the timestamp so the Heartbeat Monitor knows it is ACTIVE (Not Idle)
        conn.last_used_at = datetime.utcnow()

        # Keep plaintext in local variables only. ORM fields remain ciphertext so an
        # autoflush can never write a decrypted secret back to the database.
        access_token = decrypt_string(conn.access_token) if conn.access_token else None
        refresh_token = decrypt_string(conn.refresh_token) if conn.refresh_token else None
        client_secret = decrypt_string(conn.oauth_client_secret) if conn.oauth_client_secret else None
        
        # 3. Handle Token Expiration (The Lazy Refresh Logic)
        if conn.auth_type == "OAuth2" and conn.expires_at and refresh_token:
            # We add a 60-second safety buffer to prevent mid-run crashes
            if datetime.utcnow() + timedelta(seconds=60) > conn.expires_at:
                logger.info(f"[OAuthManager] Token expired for {conn.platform_name}. Executing Lazy Refresh...")
                
                try:
                    if not conn.oauth_client_id or not client_secret or not conn.oauth_token_url:
                        raise ValueError(f"Missing dynamic OAuth credentials (client_id, client_secret, token_url) for {conn.platform_name}")
                    validate_credential_destination(conn.oauth_token_url)
                        
                    refresh_payload = {
                        "grant_type": "refresh_token",
                        "refresh_token": refresh_token,
                        "client_id": conn.oauth_client_id,
                        "client_secret": client_secret
                    }
                    async with httpx.AsyncClient() as client:
                        response = await client.post(conn.oauth_token_url, data=refresh_payload)
                        response.raise_for_status()
                        new_data = response.json()
                        
                        access_token = new_data.get("access_token")
                        if not access_token:
                            raise ValueError("OAuth provider response did not include an access token")
                        conn.access_token = encrypt_string(access_token)
                        if new_data.get("refresh_token"):
                            refresh_token = new_data["refresh_token"]
                            conn.refresh_token = encrypt_string(refresh_token)
                        expires_in = new_data.get("expires_in", 3600)
                        conn.expires_at = datetime.utcnow() + timedelta(seconds=expires_in)
                    
                    logger.info(f"[OAuthManager] Successfully refreshed token for {conn.platform_name}.")
                except Exception as e:
                    logger.error("[OAuthManager] Failed to refresh token for %s", conn.platform_name)
                    raise ValueError(f"Failed to refresh OAuth token for {conn.platform_name}") from e

        # Transparently migrate legacy plaintext rows when they are used.
        if conn.access_token and not is_encrypted(conn.access_token):
            conn.access_token = encrypt_string(access_token)
        if conn.refresh_token and not is_encrypted(conn.refresh_token):
            conn.refresh_token = encrypt_string(refresh_token)
        if conn.oauth_client_secret and not is_encrypted(conn.oauth_client_secret):
            conn.oauth_client_secret = encrypt_string(client_secret)

        # 4. Save any timestamp or new token updates to the DB
        await db.commit()
        
        if not access_token:
            raise ValueError(f"Connection profile '{connection_id}' has no access token.")
        return access_token
