from typing import Any, Dict, Optional, Union
from pydantic import BaseModel, Field, ConfigDict
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
from abc import ABC
from langchain.tools import BaseTool

class HTTPRequestConfig(BaseModel):
    """HTTP リクエストの設定"""
    model_config = ConfigDict(frozen=True)

    timeout: float = Field(default=30.0, description="リクエストのタイムアウト時間（秒）")
    retry_attempts: int = Field(default=3, description="リトライ回数")
    retry_min_wait: float = Field(default=1.0, description="リトライ時の最小待機時間（秒）")
    retry_max_wait: float = Field(default=10.0, description="リトライ時の最大待機時間（秒）")
    headers: Dict[str, str] = Field(default_factory=dict, description="追加のHTTPヘッダー")

class HTTPResponse(BaseModel):
    """HTTP レスポンスのラッパー"""
    status_code: int = Field(..., description="HTTPステータスコード")
    data: Any = Field(..., description="レスポンスデータ")
    headers: Dict[str, str] = Field(default_factory=dict, description="レスポンスヘッダー")

class HTTPToolError(Exception):
    """HTTP ツールのエラー"""
    def __init__(self, message: str, status_code: Optional[int] = None, response: Optional[Any] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response

class HTTPBaseTool(BaseTool, ABC):
    """HTTP ベースのツールの基底クラス"""
    
    def __init__(
        self,
        endpoint: str,
        config: Optional[HTTPRequestConfig] = None,
        name: Optional[str] = None,
        description: Optional[str] = None
    ):
        """
        Args:
            endpoint: ツールのエンドポイントURL
            config: HTTPリクエストの設定
            name: ツールの名前
            description: ツールの説明
        """
        super().__init__(name=name, description=description)
        self.endpoint = endpoint
        self.config = config or HTTPRequestConfig()
        self._client = httpx.AsyncClient(
            timeout=self.config.timeout,
            headers=self.config.headers
        )

    async def _make_request(
        self,
        method: str,
        payload: Optional[Dict[str, Any]] = None
    ) -> HTTPResponse:
        """HTTP リクエストを実行

        Args:
            method: HTTPメソッド
            payload: リクエストボディ

        Returns:
            HTTPResponse: レスポンス情報

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
                response = await self._client.request(
                    method,
                    self.endpoint,
                    json=payload
                )
                response.raise_for_status()
                
                return HTTPResponse(
                    status_code=response.status_code,
                    data=response.json(),
                    headers=dict(response.headers)
                )
            except httpx.HTTPStatusError as e:
                raise HTTPToolError(
                    f"HTTP error: {str(e)}",
                    status_code=e.response.status_code,
                    response=e.response.json() if e.response.content else None
                )
            except httpx.RequestError as e:
                raise HTTPToolError(f"Request error: {str(e)}")

        return await _execute_request()

    async def arun(self, tool_input: Union[str, Dict[str, Any]], **kwargs) -> Any:
        """ツールを非同期実行

        Args:
            tool_input: ツールへの入力
            **kwargs: 追加のパラメータ

        Returns:
            Any: ツールの実行結果
        """
        payload = tool_input if isinstance(tool_input, dict) else {"input": tool_input}
        response = await self._make_request("POST", payload)
        return response.data

    def run(self, tool_input: Union[str, Dict[str, Any]], **kwargs) -> Any:
        """ツールを同期実行（非推奨）

        Note:
            可能な限り arun() の使用を推奨
        """
        import asyncio
        return asyncio.run(self.arun(tool_input, **kwargs))

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._client.aclose()