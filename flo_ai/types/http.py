# flo_ai/types/http.py
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, HttpUrl
from datetime import datetime
from uuid import UUID
from enum import Enum

# 共通の基本型
class HTTPBaseConfig(BaseModel):
    """HTTP基本設定"""
    timeout: int = 30
    retry_count: int = 3
    retry_delay: int = 1
    headers: Optional[Dict[str, str]] = None

class HTTPMessage(BaseModel):
    """メッセージの基本型"""
    content: str
    role: str
    name: Optional[str] = None
    additional_kwargs: Dict[str, Any] = Field(default_factory=dict)

class HTTPMetadata(BaseModel):
    """メタデータの基本型"""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    session_id: Optional[UUID] = None
    trace_id: Optional[str] = None

# Tool関連の型
class HTTPToolConfig(HTTPBaseConfig):
    """HTTPツール設定"""
    base_url: HttpUrl
    tool_type: str

class HTTPToolRequest(BaseModel):
    """HTTPツールリクエスト"""
    inputs: Dict[str, Any]
    metadata: HTTPMetadata

class HTTPToolResponse(BaseModel):
    """HTTPツールレスポンス"""
    outputs: Dict[str, Any]
    error: Optional[str] = None
    metadata: HTTPMetadata

# Node関連の型
class HTTPNodeConfig(HTTPBaseConfig):
    """HTTPノード設定"""
    base_url: HttpUrl
    node_type: str

class HTTPNodeRequest(BaseModel):
    """HTTPノードリクエスト"""
    messages: List[HTTPMessage]
    state: Dict[str, Any]
    metadata: HTTPMetadata

class HTTPNodeResponse(BaseModel):
    """HTTPノードレスポンス"""
    output: Dict[str, Any]
    next: Optional[str] = None
    error: Optional[str] = None
    metadata: HTTPMetadata

# Router関連の型
class RouterType(str, Enum):
    """ルータータイプ"""
    SUPERVISOR = "supervisor"
    LINEAR = "linear"
    LLM = "llm"

class HTTPRouterConfig(HTTPBaseConfig):
    """HTTPルーター設定"""
    base_url: HttpUrl
    router_type: RouterType

class RouterState(BaseModel):
    """ルーターの状態"""
    messages: List[HTTPMessage]
    current_node: str
    next_node: Optional[str] = None
    loop_tracker: Dict[str, int] = Field(default_factory=dict)
    metadata: HTTPMetadata

class HTTPRouterRequest(BaseModel):
    """HTTPルーターリクエスト"""
    state: RouterState
    available_nodes: List[str]
    metadata: HTTPMetadata

class HTTPRouterResponse(BaseModel):
    """HTTPルーターレスポンス"""
    next_node: str
    error: Optional[str] = None
    metadata: HTTPMetadata

# エラー関連の型
class HTTPError(BaseModel):
    """HTTPエラー情報"""
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None

# 監視・メトリクス関連の型
class HTTPMetrics(BaseModel):
    """HTTP実行メトリクス"""
    latency_ms: float
    status_code: int
    retry_count: int
    success: bool

# セッション関連の型
class HTTPSession(BaseModel):
    """HTTPセッション情報"""
    session_id: UUID
    created_at: datetime
    last_activity: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)

# バッチ処理関連の型
class HTTPBatchRequest(BaseModel):
    """HTTPバッチリクエスト"""
    requests: List[Union[HTTPToolRequest, HTTPNodeRequest, HTTPRouterRequest]]
    metadata: HTTPMetadata

class HTTPBatchResponse(BaseModel):
    """HTTPバッチレスポンス"""
    responses: List[Union[HTTPToolResponse, HTTPNodeResponse, HTTPRouterResponse]]
    error: Optional[str] = None
    metadata: HTTPMetadata