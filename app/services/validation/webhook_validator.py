"""Webhook URL validation service."""

from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlparse
from typing import Set

from app.domain.interfaces import IWebhookValidator


class WebhookValidationError(Exception):
    """Raised when webhook URL validation fails."""
    pass


class WebhookValidator(IWebhookValidator):
    """
    Webhook URL validation service for security.
    
    Validates webhook URLs against security policies to prevent:
    - SSRF attacks
    - Access to internal networks
    - Malicious URL schemes
    """
    
    ALLOWED_SCHEMES = {"http", "https"}
    BLOCKED_HOSTS = {
        "localhost",
        "127.0.0.1",
        "::1",
        "0.0.0.0",
        "169.254.169.254",  # AWS metadata service
        "metadata.google.internal",  # GCP metadata service
    }
    
    # Private IP ranges
    PRIVATE_NETWORKS = [
        ipaddress.ip_network("10.0.0.0/8"),
        ipaddress.ip_network("172.16.0.0/12"),
        ipaddress.ip_network("192.168.0.0/16"),
        ipaddress.ip_network("127.0.0.0/8"),
        ipaddress.ip_network("169.254.0.0/16"),  # Link-local
        ipaddress.ip_network("::1/128"),  # IPv6 loopback
        ipaddress.ip_network("fc00::/7"),  # IPv6 private
        ipaddress.ip_network("fe80::/10"),  # IPv6 link-local
    ]
    
    def validate_webhook_url(self, url: str) -> None:
        """
        Validate webhook URL against security policies.
        
        Args:
            url: The webhook URL to validate
            
        Raises:
            WebhookValidationError: If URL validation fails
        """
        if not url or not isinstance(url, str):
            raise WebhookValidationError("URL is required and must be a string")
        
        # Basic URL format validation
        try:
            parsed = urlparse(url)
        except Exception as e:
            raise WebhookValidationError(f"Invalid URL format: {e}")
        
        # Check scheme
        if not self.is_allowed_scheme(parsed.scheme):
            raise WebhookValidationError(
                f"URL scheme '{parsed.scheme}' is not allowed. "
                f"Allowed schemes: {', '.join(self.ALLOWED_SCHEMES)}"
            )
        
        # Check host
        hostname = parsed.hostname
        if not hostname:
            raise WebhookValidationError("URL must have a valid hostname")
        
        # Check if host is blocked
        if self.is_blocked_host(hostname):
            raise WebhookValidationError(f"Host '{hostname}' is blocked")
        
        # Check for private IP addresses
        try:
            ip = ipaddress.ip_address(hostname)
            if self.is_private_ip(str(ip)):
                raise WebhookValidationError(f"Private IP addresses are not allowed: {ip}")
        except ValueError:
            # Not an IP address, continue with hostname validation
            pass
        
        # Additional hostname validation
        if self._is_suspicious_hostname(hostname):
            raise WebhookValidationError(f"Suspicious hostname: {hostname}")
        
        # URL length check
        if len(url) > 2048:
            raise WebhookValidationError("URL is too long (max 2048 characters)")
    
    def is_allowed_scheme(self, scheme: str) -> bool:
        """Check if URL scheme is allowed."""
        return scheme.lower() in self.ALLOWED_SCHEMES
    
    def is_blocked_host(self, host: str) -> bool:
        """Check if host/domain is blocked."""
        return host.lower() in self.BLOCKED_HOSTS
    
    def is_private_ip(self, ip: str) -> bool:
        """Check if IP address is private/internal."""
        try:
            ip_addr = ipaddress.ip_address(ip)
            return any(ip_addr in network for network in self.PRIVATE_NETWORKS)
        except ValueError:
            return False
    
    def _is_suspicious_hostname(self, hostname: str) -> bool:
        """Check for suspicious hostname patterns."""
        # Check for suspicious patterns
        suspicious_patterns = [
            r".*\.local$",  # .local domains
            r".*\.internal$",  # .internal domains
            r".*\.corp$",  # .corp domains
            r"^\d+\.\d+\.\d+\.\d+$",  # Raw IP addresses (handled separately)
        ]
        
        for pattern in suspicious_patterns:
            if re.match(pattern, hostname, re.IGNORECASE):
                return True
        
        return False


__all__ = [
    "WebhookValidator",
    "WebhookValidationError",
]