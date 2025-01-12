from typing import Any, Dict, Optional, Union, List
from pydantic import BaseModel, Field
import httpx
from flo_ai.models.flo_node import FloNode
from flo_ai.state.flo_state import TeamFloAgentState
from flo_ai.tools.http.base import HTTPRequestConfig, HTTPResponse, HTTPToolError
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
import asyncio
from dataclasses import dataclass

class NodeRequest(BaseModel):
    """ノードへのリクエストの構造"""
    messages: List[Dict[str, Any]] = Field(..., description="メッセージ履歴")
    state: Dict[str, Any] = Field(default_factory=dict, description="ノードの状態")
    config: Optional[Dict[str, Any]] = Field(default=None, description="実行時設定")

class NodeResponse(BaseModel):
    """ノードからのレスポンスの構造"""
    messages: List[Dict[str, Any]] = Field(..., description="更新されたメッセージ履歴")
    state: Dict[str, Any] = Field(..., description="更新された状態")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="レスポンスのメタデータ")

@dataclass
class NodeExecutionResult:
    """ノード実行結果"""
    messages: List[BaseMessage]
    state: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

class HTTPFloNode(FloNode):
    """HTTPベースのノード実装"""
    
    def __init__(
        self,
        endpoint: str,
        name: str,
        config: Optional[HTTPRequestConfig] = None,
        type: str = "http",
        model_name: Optional[str] = None
    ):
        """
        Args:
            endpoint: ノードのエンドポイントURL
            name: ノードの名前
            config: HTTPリクエストの設定
            type: ノードの種類
            model_name: 使用するモデル名
        """
        super().__init__(name=name, type=type)
        self.endpoint = endpoint
        self.config = config or HTTPRequestConfig()
        self.model_name = model_name
        self._client = httpx.AsyncClient(
            timeout=self.config.request_timeout,
            headers=self.config.headers
        )

    async def _make_request(
        self,
        request: NodeRequest,
    ) -> NodeResponse:
        """HTTPリクエストの実行

        Args:
            request: ノードリクエスト

        Returns:
            NodeResponse: ノードレスポンス

        Raises:
            HTTPToolError: リクエストエラー
        """
        @retry(
            stop=stop_after_attempt(self.config.retry_attempts),
            wait=wait_exponential(
                multiplier=self.config.retry_min_wait,
                min=self.config.retry_min_wait,
                max=self.config.retry_max_wait
            ),
            reraise=True
        )
        async def _execute_request():
            try:
                response = await self._client.post(
                    self.endpoint,
                    json=request.model_dump()
                )
                response.raise_for_status()
                return NodeResponse(**response.json())
            except httpx.HTTPStatusError as e:
                raise HTTPToolError(
                    f"HTTP error: {str(e)}",
                    status_code=e.response.status_code,
                    response=e.response.json() if e.response.content else None
                )
            except httpx.RequestError as e:
                raise HTTPToolError(f"Request error: {str(e)}")

        return await _execute_request()

    def _serialize_messages(
        self, 
        messages: List[BaseMessage]
    ) -> List[Dict[str, Any]]:
        """メッセージのシリアライズ"""
        return [
            {
                "type": message.__class__.__name__,
                "content": message.content,
                "additional_kwargs": message.additional_kwargs
            }
            for message in messages
        ]

    def _deserialize_messages(
        self, 
        messages: List[Dict[str, Any]]
    ) -> List[BaseMessage]:
        """メッセージのデシリアライズ"""
        message_map = {
            "AIMessage": AIMessage,
            "HumanMessage": HumanMessage
        }
        
        return [
            message_map[msg["type"]](
                content=msg["content"],
                additional_kwargs=msg.get("additional_kwargs", {})
            )
            for msg in messages
        ]

    async def ainvoke(
        self,
        state: TeamFloAgentState,
        config: Optional[Dict[str, Any]] = None
    ) -> TeamFloAgentState:
        """ノードの非同期実行

        Args:
            state: 現在の状態
            config: 実行時設定

        Returns:
            TeamFloAgentState: 更新された状態
        """
        request = NodeRequest(
            messages=self._serialize_messages(state["messages"]),
            state={k: v for k, v in state.items() if k != "messages"},
            config=config
        )

        response = await self._make_request(request)
        
        return {
            "messages": self._deserialize_messages(response.messages),
            **response.state
        }

    def invoke(
        self,
        state: TeamFloAgentState,
        config: Optional[Dict[str, Any]] = None
    ) -> TeamFloAgentState:
        """ノードの同期実行（非推奨）"""
        return asyncio.run(self.ainvoke(state, config))

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._client.aclose()

class HTTPFloNodePool:
    """HTTPノードのプール管理"""

    def __init__(self, max_concurrent: int = 5):
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def execute_node(
        self,
        node: HTTPFloNode,
        state: TeamFloAgentState,
        config: Optional[Dict[str, Any]] = None
    ) -> TeamFloAgentState:
        """ノードの実行（セマフォ制御付き）"""
        async with self._semaphore:
            return await node.ainvoke(state, config)

class HTTPFloNodeFactory:
    """HTTPノードのファクトリー"""

    @staticmethod
    def create_node(
        endpoint: str,
        name: str,
        config: Optional[HTTPRequestConfig] = None,
        type: str = "http",
        model_name: Optional[str] = None
    ) -> HTTPFloNode:
        """新しいHTTPノードの作成"""
        return HTTPFloNode(
            endpoint=endpoint,
            name=name,
            config=config,
            type=type,
            model_name=model_name
        )

    @staticmethod
    def create_pool(max_concurrent: int = 5) -> HTTPFloNodePool:
        """ノードプールの作成"""
        return HTTPFloNodePool(max_concurrent=max_concurrent)