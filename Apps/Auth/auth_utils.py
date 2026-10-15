

from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import jwt
import os
import logging
import requests  # For HTTP calls to Keycloak (public key fetch, introspection)
import urllib3  # NEW: For suppressing SSL warnings
from dotenv import load_dotenv

load_dotenv(".env")

# NEW: Suppress SSL warnings when using self-signed certificates
# Only suppresses warnings, doesn't affect security (we're already using verify=False consciously)
if not os.getenv("KEYCLOAK_VERIFY_SSL", "true").lower() == "true":
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from typing import Optional, Set  # NEW: Type hints for better code clarity
from datetime import datetime, timedelta  # NEW: For public key cache expiration tracking


logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION: Load Keycloak Settings
# ============================================================

KEYCLOAK_SERVER_URL = os.getenv("KEYCLOAK_SERVER_URL")
KEYCLOAK_REALM_NAME = os.getenv("KEYCLOAK_REALM_NAME")
KEYCLOAK_CLIENT_ID = os.getenv("KEYCLOAK_CLIENT_ID")
KEYCLOAK_PUBLIC_KEY = os.getenv("KEYCLOAK_PUBLIC_KEY", None)
KEYCLOAK_ALGORITHMS = ["RS256"]
KEYCLOAK_ISSUER = (
    f"{KEYCLOAK_SERVER_URL.rstrip('/')}/realms/{KEYCLOAK_REALM_NAME}"
    if KEYCLOAK_SERVER_URL and KEYCLOAK_REALM_NAME
    else None
)

# NEW: Introspection configuration (for high-security endpoints)
# These allow real-time token verification by calling Keycloak
ENABLE_TOKEN_INTROSPECTION = os.getenv("ENABLE_TOKEN_INTROSPECTION", "false").lower() == "true"
INTROSPECTION_CLIENT_ID = os.getenv("INTROSPECTION_CLIENT_ID", KEYCLOAK_CLIENT_ID)
INTROSPECTION_CLIENT_SECRET = os.getenv("INTROSPECTION_CLIENT_SECRET", os.getenv("KEYCLOAK_CLIENT_SECRET_KEY"))
# NEW: SSL Verification Control for Keycloak (for self-signed certificates)
# Set to 'false' if Keycloak uses self-signed certificate (development/staging)
# Set to 'true' for production with proper SSL certificate
KEYCLOAK_VERIFY_SSL = os.getenv("KEYCLOAK_VERIFY_SSL", "true").lower() == "true"



# ============================================================
# ROLE CONFIGURATION: Load Role Names from Environment
# ============================================================

MAKER_ROLE = os.getenv("MAKER_ROLE", "maker")  # Fallback to old role if not in .env
CHECKER_ROLE = os.getenv("CHECKER_ROLE", "checker")      # Fallback to old role if not in .env
ADMIN_ROLE = os.getenv("ADMIN_ROLE", "admin")              # Admin role with unrestricted access

# ============================================================
# STARTUP LOGGING: Display Configured Roles
# ============================================================

logger.info("=" * 60)
logger.info(" JWT Token Verification Configuration:")
logger.info(f"    Public Key Source: {'Environment Variable' if KEYCLOAK_PUBLIC_KEY else 'Auto-fetch from Keycloak'}")
logger.info(f"    Token Introspection: {'Enabled' if ENABLE_TOKEN_INTROSPECTION else 'Disabled'}")
logger.info(f"    SSL Verification: {'Enabled' if KEYCLOAK_VERIFY_SSL else 'Disabled (Self-signed cert)'}")  # NEW
logger.info(f"    Keycloak Server: {KEYCLOAK_SERVER_URL}")
logger.info(f"    Realm: {KEYCLOAK_REALM_NAME}")
logger.info(f"    Client ID: {KEYCLOAK_CLIENT_ID}")
logger.info("=" * 60)

# ============================================================
# HTTP BEARER SECURITY SCHEME
# ============================================================

security = HTTPBearer()


# ============================================================
# NEW: PUBLIC KEY CACHE AND TOKEN BLACKLIST
# ============================================================

class TokenCache:
    """
    WHAT THIS DOES:
    - Stores Keycloak's public key in memory (so we don't fetch it every time)
    - Stores a list of revoked/blacklisted tokens
    - Both are kept in RAM for super fast access
    
    WHY WE NEED THIS:
    - Public key rarely changes, so caching saves network calls
    - Blacklist allows manual token revocation without Keycloak introspection
    
    NOTE:
    Think of this as a notepad where we write down:
    1. Keycloak's public key (to verify tokens)
    2. List of banned tokens (tokens we want to reject)
    """
    
    def __init__(self):
        # Store the public key here once fetched
        self.public_key: Optional[str] = None
        
        # Store revoked token IDs here (set = unique list)
        self.revoked_tokens: Set[str] = set()
        
        # Track when we last fetched the public key
        self.public_key_fetched_at: Optional[datetime] = None
        
        # Refresh public key every 24 hours
        self.public_key_ttl = timedelta(hours=24)
        
    def get_public_key(self) -> str:
        """
        Get Keycloak's public key (fetch if not cached)
        
        WHAT THIS DOES:
        1. Check if we already have the public key in memory
        2. If yes and it's still fresh (< 24 hours old), use it
        3. If no or it's stale, fetch it from Keycloak
        4. Format it as PEM and store it
        
        RETURNS:
        - str: Public key in PEM format ready for JWT verification
        
        NOTE:
        This is like checking if you have someone's phone number saved.
        If you do, use it. If not, look it up and save it for next time.
        """
        
        # Check if we need to refresh the public key
        needs_refresh = (
            self.public_key is None or 
            self.public_key_fetched_at is None or
            datetime.now() - self.public_key_fetched_at > self.public_key_ttl
        )
        
        if needs_refresh:
            logger.info(" Fetching public key from Keycloak...")
            
            try:
                # Keycloak exposes its public key at this endpoint
                realm_info_url = f"{KEYCLOAK_SERVER_URL}/realms/{KEYCLOAK_REALM_NAME}"
                
               # Make HTTP request to get realm information
                # verify=KEYCLOAK_VERIFY_SSL allows self-signed certificates if needed
                response = requests.get(
                    realm_info_url, 
                    timeout=5,
                    verify=KEYCLOAK_VERIFY_SSL  # NEW: Control SSL verification
                )
                response.raise_for_status()  # Raise error if request fails
                
                # Extract the public key from JSON response
                realm_data = response.json()
                raw_public_key = realm_data.get('public_key')
                
                if not raw_public_key:
                    raise ValueError("Public key not found in Keycloak realm data")
                
                # Format as PEM (Privacy Enhanced Mail format - standard for keys)
                # JWT libraries need this specific format
                self.public_key = (
                    "-----BEGIN PUBLIC KEY-----\n"
                    f"{raw_public_key}\n"
                    "-----END PUBLIC KEY-----"
                )
                
                # Record when we fetched it
                self.public_key_fetched_at = datetime.now()
                
                logger.info(" Public key fetched and cached successfully")
                
            except Exception as e:
                logger.error(f" Failed to fetch public key from Keycloak: {str(e)}")
                
                # If we have an old cached key, use it
                if self.public_key:
                    logger.warning(" Using stale cached public key")
                else:
                    raise HTTPException(
                        status_code=503,
                        detail="Unable to fetch Keycloak public key. Authentication unavailable."
                    )
        
        return self.public_key
    
    def is_token_revoked(self, token_id: str) -> bool:
        """
        Check if a token ID is in the blacklist
        
        PARAMETERS:
        - token_id (str): The 'jti' claim from JWT token
        
        RETURNS:
        - bool: True if token is revoked, False if valid
        
        NOTE:
        Like checking if someone is on a "banned users" list.
        If their name is on the list, don't let them in.
        """
        return token_id in self.revoked_tokens
    
    def revoke_token(self, token_id: str):
        """
        Add a token ID to the blacklist
        
        PARAMETERS:
        - token_id (str): The 'jti' claim from JWT token to revoke
        
        USE CASE:
        When you need to manually invalidate a specific token
        (e.g., user reports suspicious activity)
        
        NOTE:
        This is like adding someone's name to a "banned" list.
        Once added, their token won't work anymore.
        """
        self.revoked_tokens.add(token_id)
        logger.info(f"Token {token_id[:10]}... added to blacklist")
    
    def clear_revoked_tokens(self):
        """
        Clear all revoked tokens from blacklist
        
        USE CASE:
        During system maintenance or cleanup
        
        BEGINNER NOTE:
        This clears the entire banned list (use with caution!)
        """
        count = len(self.revoked_tokens)
        self.revoked_tokens.clear()
        logger.info(f" Cleared {count} tokens from blacklist")

# Create a single global instance (singleton pattern)
# This instance will be used throughout the application
token_cache = TokenCache()




# ============================================================
# FUNCTION 1: Decode and Validate JWT Token
# ============================================================


def decode_token(token: str, use_introspection: bool = False) -> dict:
    """
    Decode a JWT token and extract user information
    
    WHAT THIS DOES:
    - Takes the token string (encrypted message)
    - Verifies it's genuine using Keycloak's public key
    - Decodes it to read user information inside
    - Checks if token is blacklisted locally
    - Optionally verifies with Keycloak introspection for high-security
    
    PARAMETERS:
    - token (str): The JWT token from Authorization header
    - use_introspection (bool): If True, verify with Keycloak (slow but accurate)
    
    RETURNS:
    - dict: Decoded token with user info (username, roles, etc.)
    
    RAISES:
    - HTTPException: If token is invalid, expired, or revoked
    
    TWO VERIFICATION MODES:
    
    1. PUBLIC KEY MODE (DEFAULT - FAST):
       - Verifies signature using public key
       - Checks local blacklist
       - No network call to Keycloak
       - Fast but doesn't detect Keycloak-side revocations
    
    2. INTROSPECTION MODE (OPTIONAL - SECURE):
       - Calls Keycloak to verify token status
       - Detects server-side revocations
       - Slower due to network call
       - Use for critical operations only
    
    BEGINNER ANALOGY:
    PUBLIC KEY MODE = Check if someone's ID card looks genuine (fast)
    INTROSPECTION MODE = Call the ID office to verify it's still valid (slow but sure)
    """
    
    try:
        if not KEYCLOAK_SERVER_URL or not KEYCLOAK_REALM_NAME or not KEYCLOAK_CLIENT_ID:
            logger.error("Keycloak authentication settings are incomplete")
            raise HTTPException(
                status_code=503,
                detail="Authentication service is not configured",
            )
        # ============================================================
        # HYBRID APPROACH: Choose verification method
        # ============================================================
        
        if use_introspection and ENABLE_TOKEN_INTROSPECTION:
            # HIGH-SECURITY MODE: Ask Keycloak if token is valid
            logger.debug(" Using introspection for high-security verification")
            return introspect_token(token)
        
        # ============================================================
        # STANDARD MODE: Verify using public key (fast)
        # ============================================================
        
        # Step 1: Get or fetch public key from cache
        if KEYCLOAK_PUBLIC_KEY:
            # Use public key from .env if provided
            public_key = (
                "-----BEGIN PUBLIC KEY-----\n"
                f"{KEYCLOAK_PUBLIC_KEY}\n"
                "-----END PUBLIC KEY-----"
            )
            logger.debug(" Using public key from environment variable")
        else:
            # Auto-fetch from Keycloak
            public_key = token_cache.get_public_key()
            logger.debug(" Using auto-fetched public key from Keycloak")
        
        # Step 2: Decode and verify token signature
        decoded = jwt.decode(
            token,
            public_key,
            algorithms=KEYCLOAK_ALGORITHMS,
            issuer=KEYCLOAK_ISSUER,
            options={
                "verify_signature": True,   #  Verify token hasn't been tampered with
                "verify_exp": True,         #  Verify token hasn't expired
                "verify_aud": False,        #  Keycloak uses 'azp' not 'aud'
                "verify_iss": True,
            }
        )
        
        # Step 3: Verify token is for correct client (using 'azp' claim)
        token_azp = decoded.get("azp")
        if token_azp != KEYCLOAK_CLIENT_ID:
            logger.error(
                f" Invalid azp claim: token azp='{token_azp}', "
                f"expected='{KEYCLOAK_CLIENT_ID}'"
            )
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication token - wrong client"
            )
        
        # Step 4: Check local token blacklist
        token_id = decoded.get("jti")  # JWT ID - unique identifier for this token
        if token_id and token_cache.is_token_revoked(token_id):
            logger.warning(f" Token {token_id[:10]}... is blacklisted")
            raise HTTPException(
                status_code=401,
                detail="Token has been revoked"
            )
        
        # Success! Token is valid
        logger.debug(
            f" Token signature verified for user: {decoded.get('preferred_username', 'unknown')}"
        )
        
        return decoded
    
    # ============================================================
    # ERROR HANDLING: Different types of JWT errors
    # ============================================================
    
    except HTTPException:
        raise

    except jwt.ExpiredSignatureError:
        # Token has expired - user needs to login again or refresh token
        logger.warning(" Token has expired")
        raise HTTPException(
            status_code=401,
            detail="Token has expired. Please login again or refresh your token."
        )
    
    except jwt.InvalidSignatureError:
        # Token signature doesn't match - possibly tampered with
        logger.error(" Invalid token signature - possible tampering detected")
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token - signature verification failed"
        )
    
    except jwt.InvalidTokenError as e:
        # Token is malformed or invalid
        logger.error(f" Invalid token structure: {str(e)}")
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token format"
        )
    
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f" Unexpected error decoding token: {str(e)}")
        raise HTTPException(
            status_code=401,
            detail="Authentication failed"
        )


# ============================================================
# NEW FUNCTION: Token Introspection (High-Security Mode)
# ============================================================

def introspect_token(token: str) -> dict:
    """
    Verify token by calling Keycloak's introspection endpoint
    
    WHAT THIS DOES:
    - Sends token to Keycloak for real-time verification
    - Keycloak checks if token is valid, active, and not revoked
    - Returns token claims if valid
    - Raises error if invalid or revoked
    
    WHEN TO USE:
    - Critical operations (money transfers, admin actions)
    - When you need to detect server-side revocations immediately
    - When extra security is worth the performance cost
    
    HOW IT WORKS:
    1. Send token + client credentials to Keycloak
    2. Keycloak verifies token and checks its database
    3. Returns {"active": true/false, ...claims}
    4. If active=false, token is invalid/revoked
    
    PARAMETERS:
    - token (str): The JWT token to verify
    
    RETURNS:
    - dict: Token claims if valid
    
    RAISES:
    - HTTPException 401: If token is invalid or revoked
    
    PERFORMANCE:
    - Adds 50-200ms latency per request
    - Not recommended for high-traffic endpoints
    
    BEGINNER NOTE:
    This is like calling the DMV to verify someone's driver's license
    is still valid. Accurate but slow.
    """
    
    try:
        # Build introspection endpoint URL
        introspect_url = (
            f"{KEYCLOAK_SERVER_URL}/realms/{KEYCLOAK_REALM_NAME}"
            "/protocol/openid-connect/token/introspect"
        )
        
        logger.debug(f" Introspecting token at: {introspect_url}")
        
        # Prepare request data
        # Keycloak needs our backend service credentials to trust the request
        data = {
            'token': token,
            'client_id': INTROSPECTION_CLIENT_ID,
            'client_secret': INTROSPECTION_CLIENT_SECRET
        }
        
        # Send POST request to Keycloak
      # Send POST request to Keycloak
        # verify=KEYCLOAK_VERIFY_SSL allows self-signed certificates if needed
        response = requests.post(
            introspect_url,
            data=data,
            headers={'Content-Type': 'application/x-www-form-urlencoded'},
            timeout=5,  # 5 second timeout
            verify=KEYCLOAK_VERIFY_SSL  # NEW: Control SSL verification
        )
        
        # Check if request succeeded
        response.raise_for_status()
        
        # Parse JSON response
        result = response.json()
        
        # Check if token is active
        if not result.get('active', False):
            logger.warning(" Token introspection returned inactive status")
            raise HTTPException(
                status_code=401,
                detail="Token is not active or has been revoked"
            )
        
        # Token is valid! Return the claims
        logger.info(
            f" Token introspected successfully for user: "
            f"{result.get('preferred_username', 'unknown')}"
        )
        
        return result
    
    except requests.RequestException as e:
        # Network error or Keycloak unreachable
        logger.error(f" Token introspection failed: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail="Unable to verify token with Keycloak. Please try again."
        )
    
    except HTTPException:
        # Re-raise our own HTTP exceptions
        raise
    
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f" Unexpected error during introspection: {str(e)}")
        raise HTTPException(
            status_code=401,
            detail="Token verification failed"
        )



# ============================================================
# FUNCTION 2: Extract User Roles from Token
# ============================================================
def extract_user_roles(decoded_token: dict) -> list:
    """
    Extract roles from the decoded token

    WHAT THIS DOES:
    - Looks inside the token for the user's roles
    - Roles are in: resource_access → GenAI → roles

    TOKEN STRUCTURE:
    {
        "resource_access": {
            "GenAI": {
                "roles": ["maker"]  ← WE WANT THIS
            }
        }
    }

    PARAMETERS:
    - decoded_token (dict): The decoded JWT token

    RETURNS:
    - list: List of role names (e.g., ["maker"])

    EXAMPLE:
    Maker token returns: ["maker"]
    Checker token returns: ["checker"]
    """
    try:

        roles = (
            decoded_token
            .get("resource_access", {})
            .get(KEYCLOAK_CLIENT_ID, {})
            .get("roles", [])
        )

        logger.info(
            f"User {decoded_token.get('preferred_username')} has roles: {roles}")
        return roles

    except Exception as e:
        logger.error(f"Error extracting roles: {str(e)}")
        return []


# ============================================================
# FUNCTION 3: Get Current User (Main Dependency)
# ============================================================
def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security),
    require_introspection: bool = False
) -> dict:
    """
    Extract and validate user from Authorization header
    
    UPDATED: Now supports optional introspection for high-security endpoints
    
    WHAT THIS DOES:
    - Gets the "Authorization: Bearer <token>" from request
    - Extracts the token part
    - Decodes and validates it (using public key or introspection)
    - Extracts user roles
    - Returns user information with roles
    
    THIS IS A FASTAPI DEPENDENCY:
    Use this on endpoints that need authentication
    
    PARAMETERS:
    - credentials: Automatically injected by FastAPI Security
    - require_introspection: If True, use Keycloak introspection (slow but secure)
    
    RETURNS:
    - dict: User information including roles
    
    RAISES:
    - HTTPException 401: If token is invalid or missing
    
    USAGE:
    
    # Standard (fast):
    @router.get("/api/documents")
    def get_docs(current_user: dict = Depends(get_current_user)):
        # Fast public key verification
    
    # High-security (slow):
    @router.post("/api/transfer-money")
    def transfer(current_user: dict = Depends(lambda c: get_current_user(c, require_introspection=True))):
        # Real-time Keycloak verification
    """
    try:
        # Extract token from "Bearer <token>" format
        token = credentials.credentials
        
        # Decode and verify token (with optional introspection)
        decoded_token = decode_token(token, use_introspection=require_introspection)
        
        decoded_token["roles"] = extract_user_roles(decoded_token)
        
        return decoded_token
        
    except Exception as e:
        logger.error(f"Authentication failed: {str(e)}")
        raise HTTPException(
            status_code=401,
            detail="Not authenticated"
        )


# ============================================================
# NEW: Helper Function for Introspection Verification
# ============================================================
def get_current_user_with_introspection(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> dict:
    """
    Helper function that uses Keycloak introspection for token verification
    
    USE THIS for critical endpoints that need real-time Keycloak verification
    """
    return get_current_user(credentials, require_introspection=True)

# ============================================================
# FUNCTION 4: Check if User is Maker
# ============================================================
def is_maker(user: dict) -> bool:
    
    # Step 1: Extract roles from the decoded token
    roles = user.get("roles", [])
    
    # Step 2: Check if the configured MAKER_ROLE exists in user's roles
    is_maker_user = MAKER_ROLE in roles
    
    # Step 3: Log the check for debugging purposes
    logger.debug(
        f" Checking if user '{user.get('preferred_username', 'unknown')}' is Maker: "
        f"user_roles={roles}, required_role='{MAKER_ROLE}', result={is_maker_user}"
    )
    
    # Step 4: Return the result
    return is_maker_user


# ============================================================
# FUNCTION 5: Check if User is Checker
# ============================================================
def is_checker(user: dict) -> bool:
  
    # Step 1: Extract roles from the decoded token
    roles = user.get("roles", [])
    
    # Step 2: Check if the configured CHECKER_ROLE exists in user's roles
    is_checker_user = CHECKER_ROLE in roles
    
    # Step 3: Log the check for debugging purposes
    logger.debug(
        f" Checking if user '{user.get('preferred_username', 'unknown')}' is Checker: "
        f"user_roles={roles}, required_role='{CHECKER_ROLE}', result={is_checker_user}"
    )
    
    # Step 4: Return the result
    return is_checker_user


# ============================================================
# FUNCTION 5.5: Check if User is Admin
# ============================================================
def is_admin(user: dict) -> bool:
    """
    Check if user has Admin role
    
    WHAT THIS DOES:
    - Checks if the user has the ADMIN_ROLE
    - Admin users have unrestricted access to all endpoints
    
    PARAMETERS:
    - user (dict): Decoded token with user information
    
    RETURNS:
    - bool: True if user has Admin role, False otherwise
    
    USAGE:
    Admin role bypasses all maker/checker restrictions.
    Admins can access any endpoint without limitation.
    """
    # Step 1: Extract roles from the decoded token
    roles = user.get("roles", [])
    
    # Step 2: Check if the configured ADMIN_ROLE exists in user's roles
    is_admin_user = ADMIN_ROLE in roles
    
    # Step 3: Log the check for debugging purposes
    logger.debug(
        f" Checking if user '{user.get('preferred_username', 'unknown')}' is Admin: "
        f"user_roles={roles}, required_role='{ADMIN_ROLE}', result={is_admin_user}"
    )
    
    # Step 4: Return the result
    return is_admin_user

# ============================================================
# FUNCTION 6: Maker-Only Dependency
# ============================================================
def maker_only(current_user: dict = Depends(get_current_user)) -> dict:
    """
    Ensure user is a Maker (maker) or Admin

    WHAT THIS DOES:
    - Gets current user (validates token)
    - Checks if user is Maker OR Admin
    - If YES: Returns user info (endpoint continues)
    - If NO: Raises 403 Forbidden (endpoint stops)

    THIS IS A FASTAPI DEPENDENCY:
    Use this on endpoints that only Makers can access
    
    ADMIN BYPASS:
    Admin users have unrestricted access and can use this endpoint

    USAGE:
    @router.post("/doc-types/")
    def create_doc_type(current_user: dict = Depends(maker_only)):
        # This code only runs if user is Maker or Admin
        # current_user contains decoded token info

    PARAMETERS:
    - current_user: Automatically provided by Depends(get_current_user)

    RETURNS:
    - dict: User information (if Maker or Admin)

    RAISES:
    - HTTPException 403: If user is not Maker or Admin
    """
    # Check if the current user has Maker role OR Admin role
    if not (is_maker(current_user) or is_admin(current_user)):
        # User does NOT have Maker or Admin role - deny access
        
        logger.warning(
            f" UNAUTHORIZED ACCESS ATTEMPT: "
            f"User '{current_user.get('preferred_username', 'unknown')}' "
            f"attempted to access Maker-only endpoint without proper role. "
            f"Required role: MAKER_ROLE or ADMIN_ROLE, User roles: {current_user.get('roles', [])}"
        )
        
        # Raise 403 Forbidden error with dynamic role name from .env
        raise HTTPException(
            status_code=403,
            detail=f"Forbidden. This endpoint requires Maker or Admin role. "
                   f"Please contact your administrator if you believe you should have access."
        )

    return current_user


# ============================================================
# FUNCTION 7: Checker-Only Dependency
# ============================================================
def checker_only(current_user: dict = Depends(get_current_user)) -> dict:
    """
    Ensure user is a Checker (checker) or Admin

    WHAT THIS DOES:
    - Gets current user (validates token)
    - Checks if user is Checker OR Admin
    - If YES: Returns user info (endpoint continues)
    - If NO: Raises 403 Forbidden (endpoint stops)

    THIS IS A FASTAPI DEPENDENCY:
    Use this on endpoints that only Checkers can access
    
    ADMIN BYPASS:
    Admin users have unrestricted access and can use this endpoint

    USAGE:
    @router.put("/customer-profiles/reject/")
    def reject_profiles(current_user: dict = Depends(checker_only)):
        # This code only runs if user is Checker or Admin
        # current_user contains decoded token info

    PARAMETERS:
    - current_user: Automatically provided by Depends(get_current_user)

    RETURNS:
    - dict: User information (if Checker or Admin)

    RAISES:
    - HTTPException 403: If user is not Checker or Admin
    """
   # Check if the current user has Checker role OR Admin role
    if not (is_checker(current_user) or is_admin(current_user)):
        # User does NOT have Checker or Admin role - deny access
        
        logger.warning(
            f" UNAUTHORIZED ACCESS ATTEMPT: "
            f"User '{current_user.get('preferred_username', 'unknown')}' "
            f"attempted to access Checker-only endpoint without proper role. "
            f"Required role: CHECKER_ROLE or ADMIN_ROLE, User roles: {current_user.get('roles', [])}"
        )
        
        # The error message now shows the actual configured role name
        raise HTTPException(
            status_code=403,
            detail=f"Forbidden. This endpoint requires Checker or Admin role. "
                   f"Please contact your administrator if you believe you should have access."
        )

    return current_user


# ============================================================
# FUNCTION 8: Maker OR Checker Dependency
# ============================================================
def maker_or_checker(current_user: dict = Depends(get_current_user)) -> dict:
    """
    Ensure user is either Maker OR Checker OR Admin

    WHAT THIS DOES:
    - Gets current user (validates token)
    - Checks if user is Maker OR Checker OR Admin
    - If YES: Returns user info (endpoint continues)
    - If NO: Raises 403 Forbidden (endpoint stops)

    THIS IS A FASTAPI DEPENDENCY:
    Use this on endpoints that both Makers and Checkers can access
    
    ADMIN BYPASS:
    Admin users have unrestricted access and can use this endpoint

    USAGE:
    @router.put("/customer-profiles/update-status/")
    def update_status(current_user: dict = Depends(maker_or_checker)):
        # This code runs if user is EITHER Maker OR Checker OR Admin
        # Then use is_maker() or is_checker() to apply different logic
        if is_maker(current_user):
            # Maker-specific logic
        elif is_checker(current_user):
            # Checker-specific logic
        elif is_admin(current_user):
            # Admin can perform both maker and checker actions

    PARAMETERS:
    - current_user: Automatically provided by Depends(get_current_user)

    RETURNS:
    - dict: User information (if Maker or Checker or Admin)

    RAISES:
    - HTTPException 403: If user is neither Maker nor Checker nor Admin
    """
    if not (is_maker(current_user) or is_checker(current_user) or is_admin(current_user)):
        logger.warning(
            f"User {current_user.get('preferred_username')} attempted "
            f"protected endpoint without proper role"
        )
        raise HTTPException(
            status_code=403,
            detail="Forbidden. This endpoint requires Maker, Checker, or Admin role."
        )

    return current_user


# ============================================================
# FUNCTION 9: Admin-Only Dependency
# ============================================================
def admin_only(current_user: dict = Depends(get_current_user)) -> dict:
    """
    Ensure user has Admin role EXCLUSIVELY
    
    WHAT THIS DOES:
    - Gets current user (validates token)
    - Checks if user is Admin ONLY
    - If YES: Returns user info (endpoint continues)
    - If NO: Raises 403 Forbidden (endpoint stops)
    
    THIS IS A FASTAPI DEPENDENCY:
    Use this on endpoints that ONLY admins can access
    
    DIFFERENCE FROM OTHER DEPENDENCIES:
    - maker_only: Allows Maker OR Admin
    - checker_only: Allows Checker OR Admin
    - maker_or_checker: Allows Maker OR Checker OR Admin
    - admin_only: Allows ONLY Admin (this function)
    
    USE CASES:
    - Audit logs (security-sensitive data)
    - System configuration endpoints
    - User management endpoints
    - Any admin-exclusive functionality
    
    USAGE:
    @router.get("/logs/")
    def get_audit_logs(current_user: dict = Depends(admin_only)):
        # This code only runs if user is Admin
        # Makers and Checkers will get 403 Forbidden
    
    PARAMETERS:
    - current_user: Automatically provided by Depends(get_current_user)
    
    RETURNS:
    - dict: User information (if Admin)
    
    RAISES:
    - HTTPException 403: If user is not Admin
    """
    # Check if the current user has Admin role
    if not is_admin(current_user):
        # User does NOT have Admin role - deny access
        
        logger.warning(
            f" UNAUTHORIZED ACCESS ATTEMPT: "
            f"User '{current_user.get('preferred_username', 'unknown')}' "
            f"attempted to access Admin-only endpoint without proper role. "
            f"Required role: ADMIN_ROLE, User roles: {current_user.get('roles', [])}"
        )
        
        # Raise 403 Forbidden error
        raise HTTPException(
            status_code=403,
            detail=f"Forbidden. This endpoint requires Admin role exclusively. "
                   f"Please contact your administrator if you believe you should have access."
        )
    
    return current_user


# ============================================================
# FUNCTION 10: Get User Info (Helper)
# ============================================================
def get_user_info(user: dict) -> dict:
    """
    Extract clean user information from decoded token

    WHAT THIS DOES:
    - Takes decoded token
    - Extracts commonly needed fields
    - Returns them in a clean dictionary

    USEFUL FOR LOGGING:
    Makes it easy to log who did what action

    PARAMETERS:
    - user (dict): Decoded token

    RETURNS:
    - dict: {username, email, name, roles}

    EXAMPLE:
    user_info = get_user_info(current_user)
    logger.info(f"User {user_info['username']} created document")
    """
    return {
        "username": user.get("preferred_username", "unknown"),
        "email": user.get("email", ""),
        "name": user.get("name", ""),
        "roles": user.get("roles", [])
    }


    # ============================================================
# NEW: Token Management Helper Functions
# ============================================================

def revoke_token_manually(token: str):
    """
    Manually revoke a specific token
    
    WHAT THIS DOES:
    - Extracts token ID from the token
    - Adds it to the local blacklist
    - Future requests with this token will be rejected
    
    USE CASE:
    - User reports suspicious activity
    - Need to invalidate a specific session immediately
    - During security incidents
    
    PARAMETERS:
    - token (str): The JWT token to revoke
    
    BEGINNER NOTE:
    This is like canceling someone's access card immediately.
    They can't use it anymore even if it hasn't expired.
    
    EXAMPLE:
    suspicious_token = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9..."
    revoke_token_manually(suspicious_token)
    """
    try:
        # Decode token without verification (just to get the token ID)
        decoded = jwt.decode(token, options={"verify_signature": False})
        token_id = decoded.get("jti")
        
        if token_id:
            token_cache.revoke_token(token_id)
            logger.info(f" Manually revoked token: {token_id[:10]}...")
        else:
            logger.warning(" Token has no 'jti' claim, cannot revoke")
    
    except Exception as e:
        logger.error(f" Failed to revoke token: {str(e)}")


def get_public_key_status() -> dict:
    """
    Get information about the public key cache
    
    WHAT THIS DOES:
    - Returns current status of public key cache
    - Useful for debugging or monitoring
    
    RETURNS:
    - dict: Cache status information
    
    EXAMPLE RESPONSE:
    {
        "has_public_key": True,
        "last_fetched": "2025-01-15 10:30:45",
        "age_hours": 2.5,
        "will_refresh_in_hours": 21.5,
        "revoked_tokens_count": 5
    }
    """
    if token_cache.public_key_fetched_at:
        age = datetime.now() - token_cache.public_key_fetched_at
        age_hours = age.total_seconds() / 3600
        refresh_in_hours = 24 - age_hours
        
        return {
            "has_public_key": token_cache.public_key is not None,
            "last_fetched": token_cache.public_key_fetched_at.isoformat(),
            "age_hours": round(age_hours, 2),
            "will_refresh_in_hours": round(refresh_in_hours, 2),
            "revoked_tokens_count": len(token_cache.revoked_tokens)
        }
    else:
        return {
            "has_public_key": token_cache.public_key is not None,
            "last_fetched": "Never",
            "age_hours": 0,
            "will_refresh_in_hours": 24,
            "revoked_tokens_count": len(token_cache.revoked_tokens)
        }


# ============================================================
# STARTUP: Display Configuration and Test Public Key Fetch
# ============================================================

logger.info("=" * 60)
logger.info(" JWT Token Verification Configuration:")
logger.info(f"    Public Key Source: {'Environment Variable' if KEYCLOAK_PUBLIC_KEY else 'Auto-fetch from Keycloak'}")
logger.info(f"    Token Introspection: {'Enabled' if ENABLE_TOKEN_INTROSPECTION else 'Disabled'}")
logger.info(f"    Keycloak Server: {KEYCLOAK_SERVER_URL}")
logger.info(f"    Realm: {KEYCLOAK_REALM_NAME}")
logger.info(f"    Client ID: {KEYCLOAK_CLIENT_ID}")
logger.info("=" * 60)

# Public keys are fetched lazily on the first authenticated request. This keeps
# application startup independent of temporary Keycloak/network availability.
