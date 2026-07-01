"""HTTP client helpers with proper SSL certificate bundle."""

import os

import httpx

try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

try:
    import certifi

    _CERTIFI_CA = certifi.where()
except ImportError:
    _CERTIFI_CA = True


def _ssl_verify_setting() -> bool | str:
    """Resolve SSL verification: false for corporate proxies, else certifi bundle."""
    raw = os.environ.get("SSL_VERIFY", "true").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    custom_ca = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
    if custom_ca and os.path.isfile(custom_ca):
        return custom_ca
    return _CERTIFI_CA


def get_sync_http_client() -> httpx.Client:
    return httpx.Client(verify=_ssl_verify_setting())


def get_async_http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(verify=_ssl_verify_setting())
