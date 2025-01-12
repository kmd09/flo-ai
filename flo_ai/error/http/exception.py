# flo_ai/error/http/exceptions.py
from typing import Optional, Dict, Any
from pydantic import BaseModel

class HTTPFloError(Exception):
    """HTTP FloAI基底エラー"""
    def __init__(
        self, 
        message: str, 
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)

class HTTPConnectionError(HTTPFloError):
    """接続エラー"""
    def __init__(
        self, 
        message: str = "Failed to connect to the remote service",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, status_code=503, details=details)

class HTTPTimeoutError(HTTPFloError):
    """タイムアウトエラー"""
    def __init__(
        self, 
        message: str = "Request timed out",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, status_code=504, details=details)

class HTTPRetryExhaustedError(HTTPFloError):
    """リトライ上限到達エラー"""
    def __init__(
        self, 
        message: str = "Maximum retry attempts reached",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, status_code=503, details=details)

class HTTPInvalidResponseError(HTTPFloError):
    """不正なレスポンスエラー"""
    def __init__(
        self, 
        message: str = "Invalid response received from the server",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, status_code=502, details=details)

class HTTPAuthenticationError(HTTPFloError):
    """認証エラー"""
    def __init__(
        self, 
        message: str = "Authentication failed",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, status_code=401, details=details)

class HTTPRateLimitError(HTTPFloError):
    """レート制限エラー"""
    def __init__(
        self, 
        message: str = "Rate limit exceeded",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, status_code=429, details=details)

class HTTPNodeNotFoundError(HTTPFloError):
    """ノード未発見エラー"""
    def __init__(
        self, 
        node_name: str,
        details: Optional[Dict[str, Any]] = None
    ):
        message = f"Node '{node_name}' not found"
        super().__init__(message, status_code=404, details=details)

class HTTPToolExecutionError(HTTPFloError):
    """ツール実行エラー"""
    def __init__(
        self, 
        tool_name: str,
        message: str = "Tool execution failed",
        details: Optional[Dict[str, Any]] = None
    ):
        self.tool_name = tool_name
        details = details or {}
        details["tool_name"] = tool_name
        super().__init__(message, status_code=500, details=details)

class HTTPRouterError(HTTPFloError):
    """ルーターエラー"""
    def __init__(
        self, 
        router_name: str,
        message: str = "Router operation failed",
        details: Optional[Dict[str, Any]] = None
    ):
        self.router_name = router_name
        details = details or {}
        details["router_name"] = router_name
        super().__init__(message, status_code=500, details=details)