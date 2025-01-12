from typing import Any, Dict, Optional, List, Union
from pydantic import BaseModel, Field
import httpx
from .base import HTTPBaseTool, HTTPRequestConfig, HTTPToolError
import asyncio
from functools import partial

class HTTPToolRequest(BaseModel):
    """HTTPツールへのリクエストの構造"""
    tool_name: str = Field(..., description="ツールの名前")
    inputs: Dict[str, Any] = Field(..., description="ツールへの入力パラメータ")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="追加のメタデータ")

class HTTPToolClient(HTTPBaseTool):
    """HTTPツールのクライアント実装"""
    
    def __init__(
        self,
        endpoint: str,
        tool_name: str,
        config: Optional[HTTPRequestConfig] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            endpoint: ツールのエンドポイントURL
            tool_name: ツールの名前
            config: HTTPリクエストの設定
            description: ツールの説明
            metadata: 追加のメタデータ
        """
        super().__init__(
            endpoint=endpoint,
            config=config,
            name=tool_name,
            description=description
        )
        self.tool_name = tool_name
        self.metadata = metadata or {}
        
    async def _prepare_request(
        self,
        inputs: Dict[str, Any]
    ) -> HTTPToolRequest:
        """リクエストの準備

        Args:
            inputs: ツールへの入力

        Returns:
            HTTPToolRequest: 準備されたリクエスト
        """
        return HTTPToolRequest(
            tool_name=self.tool_name,
            inputs=inputs,
            metadata=self.metadata
        )

    async def arun(
        self,
        tool_input: Union[str, Dict[str, Any]],
        **kwargs
    ) -> Any:
        """ツールの非同期実行

        Args:
            tool_input: ツールへの入力
            **kwargs: 追加のパラメータ

        Returns:
            Any: ツールの実行結果
        """
        if isinstance(tool_input, str):
            inputs = {"input": tool_input}
        else:
            inputs = tool_input

        request = await self._prepare_request(inputs)
        response = await self._make_request("POST", request.model_dump())
        return response.data

class BatchHTTPToolClient:
    """複数のHTTPツールを一括実行するためのクライアント"""

    def __init__(self, max_concurrent: int = 5):
        """
        Args:
            max_concurrent: 同時実行する最大リクエスト数
        """
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        
    async def _execute_tool(
        self,
        tool: HTTPToolClient,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """個別ツールの実行

        Args:
            tool: 実行するツール
            input_data: 入力データ

        Returns:
            Dict[str, Any]: 実行結果
        """
        async with self._semaphore:
            try:
                result = await tool.arun(input_data)
                return {
                    "tool_name": tool.tool_name,
                    "status": "success",
                    "result": result
                }
            except HTTPToolError as e:
                return {
                    "tool_name": tool.tool_name,
                    "status": "error",
                    "error": str(e),
                    "status_code": e.status_code
                }
            except Exception as e:
                return {
                    "tool_name": tool.tool_name,
                    "status": "error",
                    "error": str(e)
                }

    async def execute_batch(
        self,
        tools: List[HTTPToolClient],
        inputs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """複数ツールの一括実行

        Args:
            tools: 実行するツールのリスト
            inputs: 各ツールへの入力データのリスト

        Returns:
            List[Dict[str, Any]]: 実行結果のリスト
        """
        if len(tools) != len(inputs):
            raise ValueError("tools and inputs must have the same length")

        tasks = [
            self._execute_tool(tool, input_data)
            for tool, input_data in zip(tools, inputs)
        ]
        
        results = await asyncio.gather(*tasks)
        return results

class HTTPToolFactory:
    """HTTPツールのファクトリークラス"""

    @staticmethod
    def create_tool(
        endpoint: str,
        tool_name: str,
        description: Optional[str] = None,
        config: Optional[HTTPRequestConfig] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> HTTPToolClient:
        """新しいHTTPツールの作成

        Args:
            endpoint: ツールのエンドポイントURL
            tool_name: ツールの名前
            description: ツールの説明
            config: HTTPリクエストの設定
            metadata: 追加のメタデータ

        Returns:
            HTTPToolClient: 作成されたツール
        """
        return HTTPToolClient(
            endpoint=endpoint,
            tool_name=tool_name,
            description=description,
            config=config,
            metadata=metadata
        )

    @staticmethod
    def create_batch_client(
        max_concurrent: int = 5
    ) -> BatchHTTPToolClient:
        """バッチクライアントの作成

        Args:
            max_concurrent: 同時実行する最大リクエスト数

        Returns:
            BatchHTTPToolClient: バッチクライアント
        """
        return BatchHTTPToolClient(max_concurrent=max_concurrent)