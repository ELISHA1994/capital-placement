"""
Local secret management implementations for development.
"""

import os
import json
from typing import Optional, List
from pathlib import Path
import structlog

from app.domain.interfaces import ISecretManager

logger = structlog.get_logger(__name__)


class LocalSecretManager(ISecretManager):
    """Local file-based secret manager for development."""
    
    def __init__(self, secrets_file: str = ".local_secrets.json"):
        self.secrets_file = Path(secrets_file)
        self._secrets = {}
        self._load_secrets()
    
    def _load_secrets(self):
        """Load secrets from file or environment variables."""
        # First load from environment variables
        env_secrets = {
            key.replace("SECRET_", "").lower(): value
            for key, value in os.environ.items()
            if key.startswith("SECRET_")
        }
        self._secrets.update(env_secrets)
        
        # Then load from file if it exists
        if self.secrets_file.exists():
            try:
                with open(self.secrets_file, 'r') as f:
                    file_secrets = json.load(f)
                    self._secrets.update(file_secrets)
                    logger.debug("Loaded secrets from file", count=len(file_secrets))
            except Exception as e:
                logger.warning("Failed to load secrets file", error=str(e))
    
    def _save_secrets(self):
        """Save secrets to file."""
        try:
            # Only save non-environment secrets to avoid exposing env vars
            file_secrets = {
                key: value for key, value in self._secrets.items()
                if not os.environ.get(f"SECRET_{key.upper()}")
            }
            
            with open(self.secrets_file, 'w') as f:
                json.dump(file_secrets, f, indent=2)
                logger.debug("Saved secrets to file", count=len(file_secrets))
        except Exception as e:
            logger.error("Failed to save secrets file", error=str(e))
    
    async def get_secret(self, secret_name: str) -> Optional[str]:
        """Get secret value."""
        # Check environment variables first with SECRET_ prefix
        env_key = f"SECRET_{secret_name.upper()}"
        env_value = os.getenv(env_key)
        if env_value:
            return env_value
        
        # Then check our local storage
        value = self._secrets.get(secret_name.lower())
        if value:
            logger.debug("Retrieved local secret", secret_name=secret_name)
            return value
        
        logger.warning("Secret not found", secret_name=secret_name)
        return None
    
    async def set_secret(self, secret_name: str, secret_value: str) -> bool:
        """Store secret value."""
        try:
            self._secrets[secret_name.lower()] = secret_value
            self._save_secrets()
            logger.info("Secret stored locally", secret_name=secret_name)
            return True
        except Exception as e:
            logger.error("Failed to store secret", secret_name=secret_name, error=str(e))
            return False
    
    async def delete_secret(self, secret_name: str) -> bool:
        """Delete secret."""
        try:
            key = secret_name.lower()
            if key in self._secrets:
                del self._secrets[key]
                self._save_secrets()
                logger.info("Secret deleted locally", secret_name=secret_name)
                return True
            return False
        except Exception as e:
            logger.error("Failed to delete secret", secret_name=secret_name, error=str(e))
            return False
    
    async def list_secrets(self) -> List[str]:
        """List available secret names."""
        # Include both file secrets and environment secrets
        env_secrets = [
            key.replace("SECRET_", "").lower()
            for key in os.environ.keys()
            if key.startswith("SECRET_")
        ]
        
        all_secrets = list(set(list(self._secrets.keys()) + env_secrets))
        logger.debug("Listed secrets", count=len(all_secrets))
        return all_secrets


class EnvironmentSecretManager(ISecretManager):
    """Simple secret manager that only uses environment variables."""
    
    async def get_secret(self, secret_name: str) -> Optional[str]:
        """Get secret from environment variables."""
        # Try multiple environment variable patterns
        patterns = [
            secret_name.upper(),
            f"SECRET_{secret_name.upper()}",
            secret_name.lower(),
            secret_name
        ]
        
        for pattern in patterns:
            value = os.getenv(pattern)
            if value:
                logger.debug("Retrieved env secret", secret_name=secret_name, pattern=pattern)
                return value
        
        logger.warning("Secret not found in environment", secret_name=secret_name)
        return None
    
    async def set_secret(self, secret_name: str, secret_value: str) -> bool:
        """Set secret in environment (runtime only)."""
        try:
            os.environ[f"SECRET_{secret_name.upper()}"] = secret_value
            logger.info("Secret set in environment", secret_name=secret_name)
            return True
        except Exception as e:
            logger.error("Failed to set env secret", secret_name=secret_name, error=str(e))
            return False
    
    async def delete_secret(self, secret_name: str) -> bool:
        """Delete secret from environment."""
        try:
            patterns = [
                secret_name.upper(),
                f"SECRET_{secret_name.upper()}",
                secret_name.lower(),
                secret_name
            ]
            
            deleted = False
            for pattern in patterns:
                if pattern in os.environ:
                    del os.environ[pattern]
                    deleted = True
            
            if deleted:
                logger.info("Secret deleted from environment", secret_name=secret_name)
            return deleted
        except Exception as e:
            logger.error("Failed to delete env secret", secret_name=secret_name, error=str(e))
            return False
    
    async def list_secrets(self) -> List[str]:
        """List secrets in environment."""
        secrets = [
            key.replace("SECRET_", "").lower()
            for key in os.environ.keys()
            if key.startswith("SECRET_")
        ]
        logger.debug("Listed env secrets", count=len(secrets))
        return secrets
