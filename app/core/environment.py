"""
Environment Detection and Service Strategy Selection
"""

import os
import importlib
from enum import Enum
from typing import Dict, Any, Optional
import structlog

logger = structlog.get_logger(__name__)


class Environment(Enum):
    """Environment types"""
    LOCAL = "local"
    DEVELOPMENT = "development" 
    STAGING = "staging"
    PRODUCTION = "production"


class ServiceAvailability:
    """Check which services are available in current environment"""
    
    def __init__(self):
        self._cache: Dict[str, bool] = {}
    
    def is_azure_available(self) -> bool:
        """Check if Azure SDK is available"""
        if "azure" not in self._cache:
            self._cache["azure"] = self._check_import("azure.identity")
        return self._cache["azure"]
    
    def is_redis_available(self) -> bool:
        """Check if Redis is available"""
        if "redis" not in self._cache:
            self._cache["redis"] = self._check_import("redis")
        return self._cache["redis"]
    
    def is_openai_available(self) -> bool:
        """Check if OpenAI is available"""
        if "openai" not in self._cache:
            self._cache["openai"] = self._check_import("openai")
        return self._cache["openai"]
    
    def _check_import(self, module_name: str) -> bool:
        """Safely check if a module can be imported"""
        try:
            importlib.import_module(module_name)
            return True
        except ImportError:
            return False


class EnvironmentDetector:
    """Intelligent environment detection"""
    
    def __init__(self):
        self.availability = ServiceAvailability()
    
    def detect_environment(self) -> Environment:
        """Detect current environment"""
        
        # Check explicit environment variable first
        env_var = os.getenv("ENVIRONMENT", "").lower()
        if env_var in ["local", "development", "staging", "production"]:
            return Environment(env_var)
        
        # Auto-detect based on available services and environment indicators
        if self._is_azure_environment():
            if os.getenv("AZURE_FUNCTIONS_ENVIRONMENT"):
                return Environment.PRODUCTION
            elif os.getenv("APPSETTING_WEBSITE_SITE_NAME"):
                return Environment.PRODUCTION
            else:
                return Environment.STAGING
        
        elif self._is_local_development():
            return Environment.LOCAL
        
        else:
            return Environment.DEVELOPMENT
    
    def _is_azure_environment(self) -> bool:
        """Check if running in Azure"""
        azure_indicators = [
            "AZURE_CLIENT_ID",
            "MSI_ENDPOINT", 
            "IDENTITY_ENDPOINT",
            "WEBSITE_SITE_NAME",
            "AZURE_FUNCTIONS_ENVIRONMENT"
        ]
        
        return any(os.getenv(indicator) for indicator in azure_indicators)
    
    def _is_local_development(self) -> bool:
        """Check if running in local development"""
        local_indicators = [
            os.getenv("DEBUG") == "true",
            os.getenv("ENVIRONMENT") in ["local", "dev"],
            not self.availability.is_azure_available(),
            os.path.exists(".env"),
            os.getcwd().endswith("capital-placement")
        ]
        
        return any(local_indicators)
    
    def get_service_strategy(self) -> "ServiceStrategy":
        """Get appropriate service strategy for environment"""
        env = self.detect_environment()
        
        if env == Environment.LOCAL:
            return LocalServiceStrategy()
        elif env in [Environment.DEVELOPMENT, Environment.STAGING]:
            return HybridServiceStrategy()
        else:
            return AzureServiceStrategy()


class ServiceStrategy:
    """Base strategy for service selection"""
    
    def should_use_azure_storage(self) -> bool:
        return False
    
    def should_use_azure_search(self) -> bool:
        return False
    
    def should_use_azure_openai(self) -> bool:
        return False
    
    def should_use_redis(self) -> bool:
        return True
    
    def get_fallback_services(self) -> Dict[str, str]:
        """Get fallback service implementations"""
        return {}


class LocalServiceStrategy(ServiceStrategy):
    """Local development strategy - use local alternatives"""
    
    def should_use_redis(self) -> bool:
        detector = ServiceAvailability()
        return detector.is_redis_available()
    
    def get_fallback_services(self) -> Dict[str, str]:
        return {
            "database": "postgresql",
            "storage": "local_filesystem", 
            "search": "local_search",
            "openai": "mock_ai",
            "cache": "memory_cache" if not self.should_use_redis() else "redis"
        }


class HybridServiceStrategy(ServiceStrategy):
    """Hybrid strategy - use Azure where available, local fallbacks"""
    
    def __init__(self):
        self.availability = ServiceAvailability()
    
# Cosmos DB removed - using PostgreSQL only
    
    def should_use_azure_storage(self) -> bool:
        return self.availability.is_azure_available() and bool(os.getenv("AZURE_STORAGE_ACCOUNT_NAME"))
    
    def should_use_azure_search(self) -> bool:
        return self.availability.is_azure_available() and bool(os.getenv("AZURE_SEARCH_ENDPOINT"))
    
    def should_use_azure_openai(self) -> bool:
        return self.availability.is_openai_available() and bool(os.getenv("AZURE_OPENAI_ENDPOINT"))


class AzureServiceStrategy(ServiceStrategy):
    """Azure production strategy - use all Azure services"""
    
    # Cosmos DB removed - using PostgreSQL only
    
    def should_use_azure_storage(self) -> bool:
        return True
    
    def should_use_azure_search(self) -> bool:
        return True
    
    def should_use_azure_openai(self) -> bool:
        return True


# Singleton instances
_detector: Optional[EnvironmentDetector] = None


def get_environment_detector() -> EnvironmentDetector:
    """Get singleton environment detector"""
    global _detector
    if _detector is None:
        _detector = EnvironmentDetector()
    return _detector


def get_current_environment() -> Environment:
    """Get current environment"""
    return get_environment_detector().detect_environment()


def get_service_strategy() -> ServiceStrategy:
    """Get current service strategy"""
    return get_environment_detector().get_service_strategy()


def log_environment_info():
    """Log environment information for debugging"""
    detector = get_environment_detector()
    env = detector.detect_environment()
    strategy = detector.get_service_strategy()
    
    logger.info("Environment detected",
        environment=env.value,
        strategy=strategy.__class__.__name__,
        azure_available=detector.availability.is_azure_available(),
        redis_available=detector.availability.is_redis_available(),
        openai_available=detector.availability.is_openai_available()
    )