"""
OAuth Token Validator

Enhanced OAuth 2.1 token validation with PKCE support, token introspection,
and refresh token handling.
"""

import base64
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, List

from jose import JWTError, jwt
from pydantic import BaseModel

from observability.logging_config import get_logger
from proxy.config import get_proxy_config

logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------


class TokenPayload(BaseModel):
    """JWT token payload."""

    sub: str  # Subject (user ID)
    exp: datetime  # Expiration
    iat: datetime  # Issued at
    jti: Optional[str] = None  # JWT ID
    client_id: Optional[str] = None
    scopes: List[str] = []


class User(BaseModel):
    """Authenticated user."""

    id: str
    client_id: Optional[str] = None
    scopes: List[str] = []


class TokenIntrospectionResponse(BaseModel):
    """RFC 7662 Token Introspection Response."""
    
    active: bool
    scope: Optional[str] = None
    client_id: Optional[str] = None
    username: Optional[str] = None
    token_type: Optional[str] = None
    exp: Optional[int] = None
    iat: Optional[int] = None
    nbf: Optional[int] = None
    sub: Optional[str] = None
    aud: Optional[str] = None
    iss: Optional[str] = None
    jti: Optional[str] = None


class TokenResponse(BaseModel):
    """OAuth token response."""
    
    access_token: str
    token_type: str = "Bearer"
    expires_in: int
    refresh_token: Optional[str] = None
    scope: Optional[str] = None


class PKCEVerificationError(Exception):
    """PKCE verification failed."""
    pass


# -----------------------------------------------------------------------------
# OAuth Validator
# -----------------------------------------------------------------------------


class OAuthValidator:
    """
    OAuth 2.1 token validator with PKCE support.

    Features:
    - JWT access token validation
    - PKCE verification (S256 and plain methods)
    - Token introspection (RFC 7662)
    - Refresh token coordination with RefreshTokenStore
    """

    def __init__(self):
        self.config = get_proxy_config()

    async def validate_token(self, authorization: Optional[str]) -> User:
        """
        Validate authorization header and extract user.

        Args:
            authorization: Bearer token from Authorization header

        Returns:
            User object with validated claims

        Raises:
            PermissionError: If token is invalid or expired
        """
        if not authorization:
            raise PermissionError("Missing authorization header")

        # Extract token from "Bearer <token>"
        try:
            scheme, token = authorization.split()
            if scheme.lower() != "bearer":
                raise PermissionError("Invalid authorization scheme")
        except ValueError:
            raise PermissionError("Invalid authorization header format")

        # Decode and validate JWT
        try:
            payload = jwt.decode(
                token,
                self.config.jwt_secret_key,
                algorithms=[self.config.jwt_algorithm],
            )

            token_data = TokenPayload(
                sub=payload.get("sub"),
                exp=datetime.fromtimestamp(payload.get("exp")),
                iat=datetime.fromtimestamp(payload.get("iat")),
                jti=payload.get("jti"),
                client_id=payload.get("client_id"),
                scopes=payload.get("scopes", []),
            )

            # Check expiration
            if token_data.exp < datetime.utcnow():
                raise PermissionError("Token has expired")

            return User(
                id=token_data.sub,
                client_id=token_data.client_id,
                scopes=token_data.scopes,
            )

        except JWTError as e:
            logger.warning("jwt_validation_failed", error=str(e))
            raise PermissionError(f"Invalid token: {e}")

    def create_token(
        self,
        user_id: str,
        client_id: Optional[str] = None,
        scopes: Optional[List[str]] = None,
        jti: Optional[str] = None,
    ) -> str:
        """
        Create a new JWT access token.

        Args:
            user_id: User identifier
            client_id: Client application ID
            scopes: Access scopes
            jti: Optional JWT ID (generated if not provided)

        Returns:
            Encoded JWT token
        """
        now = datetime.utcnow()
        expires = now + timedelta(hours=self.config.jwt_expiration_hours)

        payload = {
            "sub": user_id,
            "exp": expires,
            "iat": now,
            "jti": jti or secrets.token_urlsafe(16),
            "client_id": client_id,
            "scopes": scopes or [],
        }

        token = jwt.encode(
            payload,
            self.config.jwt_secret_key,
            algorithm=self.config.jwt_algorithm,
        )

        logger.info("token_created", user_id=user_id, client_id=client_id)
        return token

    def create_token_response(
        self,
        user_id: str,
        client_id: Optional[str] = None,
        scopes: Optional[List[str]] = None,
        include_refresh_token: bool = False,
    ) -> TokenResponse:
        """
        Create a full token response with access token.
        
        Note: For refresh tokens, use RefreshTokenStore directly.
        """
        access_token = self.create_token(user_id, client_id, scopes)
        
        return TokenResponse(
            access_token=access_token,
            token_type="Bearer",
            expires_in=self.config.jwt_expiration_hours * 3600,
            scope=" ".join(scopes) if scopes else None,
        )

    def introspect_token(self, token: str) -> TokenIntrospectionResponse:
        """
        Introspect an access token (RFC 7662).
        
        Args:
            token: The access token to introspect
            
        Returns:
            Token introspection response
        """
        try:
            payload = jwt.decode(
                token,
                self.config.jwt_secret_key,
                algorithms=[self.config.jwt_algorithm],
            )
            
            exp_timestamp = payload.get("exp")
            iat_timestamp = payload.get("iat")
            
            # Check if expired
            if exp_timestamp and datetime.fromtimestamp(exp_timestamp) < datetime.utcnow():
                return TokenIntrospectionResponse(active=False)
            
            scopes = payload.get("scopes", [])
            
            return TokenIntrospectionResponse(
                active=True,
                scope=" ".join(scopes) if scopes else None,
                client_id=payload.get("client_id"),
                username=payload.get("sub"),
                token_type="Bearer",
                exp=exp_timestamp,
                iat=iat_timestamp,
                sub=payload.get("sub"),
                jti=payload.get("jti"),
            )
            
        except JWTError:
            return TokenIntrospectionResponse(active=False)

    # -------------------------------------------------------------------------
    # PKCE Support
    # -------------------------------------------------------------------------
    
    @staticmethod
    def generate_code_verifier() -> str:
        """Generate a PKCE code verifier (43-128 chars)."""
        return secrets.token_urlsafe(64)[:128]
    
    @staticmethod
    def generate_code_challenge(code_verifier: str, method: str = "S256") -> str:
        """
        Generate a PKCE code challenge from a verifier.
        
        Args:
            code_verifier: The code verifier
            method: "S256" (recommended) or "plain"
            
        Returns:
            The code challenge
        """
        if method == "plain":
            return code_verifier
        elif method == "S256":
            digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
            return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
        else:
            raise ValueError(f"Unsupported PKCE method: {method}")
    
    @staticmethod
    def verify_pkce_challenge(
        code_verifier: str,
        code_challenge: str,
        method: str = "S256",
    ) -> bool:
        """
        Verify a PKCE code verifier against a challenge.
        
        Args:
            code_verifier: The code verifier from the token request
            code_challenge: The code challenge from the authorization request
            method: The challenge method ("S256" or "plain")
            
        Returns:
            True if verification succeeds
            
        Raises:
            PKCEVerificationError: If verification fails
        """
        if method == "plain":
            if code_verifier != code_challenge:
                raise PKCEVerificationError("PKCE verification failed: plain mismatch")
            return True
        elif method == "S256":
            expected_challenge = OAuthValidator.generate_code_challenge(code_verifier, "S256")
            if expected_challenge != code_challenge:
                raise PKCEVerificationError("PKCE verification failed: S256 mismatch")
            return True
        else:
            raise PKCEVerificationError(f"Unsupported PKCE method: {method}")
    
    async def create_authorization_code(
        self,
        client_id: str,
        user_id: str,
        scopes: List[str],
        redirect_uri: str,
        code_challenge: str,
        code_challenge_method: str = "S256",
        session: "AsyncSession" = None,
    ) -> str:
        """
        Create an authorization code for PKCE flow.
        
        Args:
            client_id: Client application ID
            user_id: Authenticated user ID
            scopes: Requested scopes
            redirect_uri: Redirect URI for callback
            code_challenge: PKCE code challenge
            code_challenge_method: PKCE method (S256 or plain)
            session: Database session
            
        Returns:
            The authorization code
        """
        from database.schema import PKCEChallenge
        
        code = secrets.token_urlsafe(32)
        expires_at = datetime.utcnow() + timedelta(minutes=10)  # 10 min expiry
        
        challenge = PKCEChallenge(
            code=code,
            code_challenge=code_challenge,
            code_challenge_method=code_challenge_method,
            client_id=client_id,
            user_id=user_id,
            scopes=",".join(scopes),
            redirect_uri=redirect_uri,
            expires_at=expires_at,
        )
        
        session.add(challenge)
        await session.commit()
        
        logger.info(
            "authorization_code_created",
            client_id=client_id,
            user_id=user_id,
            method=code_challenge_method,
        )
        
        return code
    
    async def exchange_authorization_code(
        self,
        code: str,
        code_verifier: str,
        client_id: str,
        redirect_uri: str,
        session: "AsyncSession" = None,
    ) -> Optional[TokenResponse]:
        """
        Exchange an authorization code for tokens (PKCE flow).
        
        Args:
            code: The authorization code
            code_verifier: The PKCE code verifier
            client_id: Client application ID
            redirect_uri: Redirect URI (must match original)
            session: Database session
            
        Returns:
            Token response or None if invalid
        """
        from sqlalchemy import select
        from database.schema import PKCEChallenge
        
        # Find the authorization code
        query = select(PKCEChallenge).where(
            PKCEChallenge.code == code,
            PKCEChallenge.client_id == client_id,
            PKCEChallenge.used_at.is_(None),
            PKCEChallenge.expires_at > datetime.utcnow(),
        )
        
        result = await session.execute(query)
        challenge = result.scalar_one_or_none()
        
        if not challenge:
            logger.warning(
                "authorization_code_exchange_failed",
                reason="code_not_found_or_expired",
            )
            return None
        
        # Verify redirect URI
        if challenge.redirect_uri != redirect_uri:
            logger.warning(
                "authorization_code_exchange_failed",
                reason="redirect_uri_mismatch",
            )
            return None
        
        # Verify PKCE
        try:
            self.verify_pkce_challenge(
                code_verifier,
                challenge.code_challenge,
                challenge.code_challenge_method,
            )
        except PKCEVerificationError as e:
            logger.warning(
                "authorization_code_exchange_failed",
                reason=str(e),
            )
            return None
        
        # Mark code as used
        challenge.used_at = datetime.utcnow()
        await session.commit()
        
        # Create token response
        scopes = challenge.scopes.split(",") if challenge.scopes else []
        
        logger.info(
            "authorization_code_exchanged",
            client_id=client_id,
            user_id=challenge.user_id,
        )
        
        return self.create_token_response(
            user_id=challenge.user_id,
            client_id=client_id,
            scopes=scopes,
        )


# Singleton instance
oauth_validator = OAuthValidator()
