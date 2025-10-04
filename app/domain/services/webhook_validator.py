"""
Domain service for webhook URL validation with SSRF protection.

This service implements security checks to prevent Server-Side Request Forgery (SSRF) attacks
and ensures webhook URLs comply with organizational security policies.
"""

import ipaddress
import re
from typing import Set
from urllib.parse import urlparse, ParseResult

from app.domain.interfaces import IWebhookValidator
from app.domain.exceptions import WebhookValidationError


class WebhookValidator(IWebhookValidator):
    """
    Webhook URL validator with comprehensive SSRF protection.
    
    This validator implements security measures to prevent webhook URLs from:
    - Targeting internal/private IP addresses
    - Using unsafe URL schemes
    - Accessing blocked domains
    - Using malformed URLs
    """

    # Allowed URL schemes for webhook URLs
    ALLOWED_SCHEMES: Set[str] = {"http", "https"}
    
    # Blocked domains (can be extended based on security requirements)
    BLOCKED_DOMAINS: Set[str] = {
        "localhost",
        "0.0.0.0",
        "127.0.0.1",
        "::1",
        "metadata.google.internal",  # GCP metadata service
        "169.254.169.254",  # AWS/Azure metadata service
        "kubernetes.default.svc",  # Kubernetes API
        "consul.service.consul",  # Consul service discovery
    }
    
    # Private IP address ranges (RFC 1918, RFC 4193, etc.)
    PRIVATE_IP_RANGES = [
        ipaddress.ip_network("10.0.0.0/8"),      # RFC 1918
        ipaddress.ip_network("172.16.0.0/12"),   # RFC 1918
        ipaddress.ip_network("192.168.0.0/16"),  # RFC 1918
        ipaddress.ip_network("127.0.0.0/8"),     # Loopback
        ipaddress.ip_network("169.254.0.0/16"),  # Link-local
        ipaddress.ip_network("::1/128"),         # IPv6 loopback
        ipaddress.ip_network("fc00::/7"),        # IPv6 unique local
        ipaddress.ip_network("fe80::/10"),       # IPv6 link-local
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
            raise WebhookValidationError("Webhook URL must be a non-empty string")
        
        # Strip whitespace
        url = url.strip()
        
        if not url:
            raise WebhookValidationError("Webhook URL cannot be empty")
        
        # Parse URL
        try:
            parsed_url = urlparse(url)
        except Exception as e:
            raise WebhookValidationError(f"Invalid URL format: {e}")
        
        # Validate URL components
        self._validate_url_components(parsed_url)
        self._validate_scheme(parsed_url.scheme)
        self._validate_hostname(parsed_url.hostname)
        
    def _validate_url_components(self, parsed_url: ParseResult) -> None:
        """Validate basic URL components."""
        if not parsed_url.scheme:
            raise WebhookValidationError("URL must include a scheme (http or https)")
        
        if not parsed_url.netloc:
            raise WebhookValidationError("URL must include a hostname")
        
        # Check for suspicious patterns
        if any(char in parsed_url.geturl() for char in ['\n', '\r', '\t']):
            raise WebhookValidationError("URL contains invalid characters")
        
        # Prevent URL encoding bypass attempts
        if '%' in parsed_url.netloc:
            # Allow only standard port encoding like %3A for :
            if not re.match(r'^[a-zA-Z0-9\-\.%:]+$', parsed_url.netloc):
                raise WebhookValidationError("URL contains suspicious encoding")

    def _validate_scheme(self, scheme: str) -> None:
        """Validate URL scheme."""
        if not self.is_allowed_scheme(scheme):
            raise WebhookValidationError(
                f"URL scheme '{scheme}' not allowed. Only {', '.join(self.ALLOWED_SCHEMES)} are permitted"
            )

    def _validate_hostname(self, hostname: str) -> None:
        """Validate hostname for security issues."""
        if not hostname:
            raise WebhookValidationError("URL must include a valid hostname")
        
        # Normalize hostname
        hostname = hostname.lower().strip()
        
        # Check blocked domains
        if self.is_blocked_host(hostname):
            raise WebhookValidationError(f"Hostname '{hostname}' is not allowed")
        
        # Check for IP addresses
        try:
            ip = ipaddress.ip_address(hostname)
            if self.is_private_ip(str(ip)):
                raise WebhookValidationError(f"Private IP address '{hostname}' is not allowed")
        except ValueError:
            # Not an IP address, continue with hostname validation
            pass
        
        # Additional hostname security checks
        self._validate_hostname_patterns(hostname)

    def _validate_hostname_patterns(self, hostname: str) -> None:
        """Check for suspicious hostname patterns."""
        # Prevent homograph attacks and suspicious patterns
        suspicious_patterns = [
            r'\.\.+',  # Multiple consecutive dots
            r'^\.',    # Starting with dot
            r'\.$',    # Ending with dot
            r'[^\x00-\x7F]',  # Non-ASCII characters
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, hostname):
                raise WebhookValidationError(f"Hostname contains suspicious pattern: {pattern}")
        
        # Check for excessive subdomain nesting (potential DNS rebinding)
        parts = hostname.split('.')
        if len(parts) > 10:
            raise WebhookValidationError("Hostname has too many subdomain levels")

    def is_allowed_scheme(self, scheme: str) -> bool:
        """Check if URL scheme is allowed."""
        return scheme.lower() in self.ALLOWED_SCHEMES

    def is_blocked_host(self, host: str) -> bool:
        """Check if host/domain is blocked."""
        host = host.lower().strip()
        
        # Direct match
        if host in self.BLOCKED_DOMAINS:
            return True
        
        # Check for subdomain matches of blocked domains
        for blocked_domain in self.BLOCKED_DOMAINS:
            if host.endswith(f'.{blocked_domain}'):
                return True
        
        return False

    def is_private_ip(self, ip: str) -> bool:
        """Check if IP address is private/internal."""
        try:
            ip_addr = ipaddress.ip_address(ip)
            for network in self.PRIVATE_IP_RANGES:
                if ip_addr in network:
                    return True
            return False
        except ValueError:
            # Not a valid IP address
            return False


__all__ = ["WebhookValidator"]