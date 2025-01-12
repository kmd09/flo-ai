# flo_ai/tools/flo_tool.py
import asyncio
from typing import Optional, Any, Type, Union, Dict, Callable
from langchain.tools import tool, BaseTool
from functools import wraps
from pydantic import BaseModel
import httpx

class FloToolConfig(BaseModel):
    """FloTool の設定"""
    handle_errors: bool = True  # エラーハンドリングを有効にするかどうか
    max_retries: int = 3       # リトライ回数
    timeout: int = 30          # タイムアウト時間（秒）

class HTTPToolConfig(FloToolConfig):
    """HTTP FloTool の設定"""
    base_url: str
    headers: Optional[Dict[str, str]] = None

def flotool(
    name: str,
    description: Optional[str] = None,
    argument_contract: Optional[type] = None,
    config: Optional[Union[FloToolConfig, HTTPToolConfig]] = None,
):
    """FloTool デコレータ

    Args:
        name (str): ツールの名前
        description (Optional[str], optional): ツールの説明
        argument_contract (Optional[type], optional): 引数のスキーマ
        config (Optional[Union[FloToolConfig, HTTPToolConfig]], optional): ツールの設定
    """
    def decorator(func_or_url: Union[Callable, str]):
        # デフォルト設定
        tool_config = config or FloToolConfig()

        # HTTP URLが渡された場合、HTTPツールを作成
        if isinstance(func_or_url, str):
            http_config = HTTPToolConfig(
                base_url=func_or_url,
                handle_errors=tool_config.handle_errors,
                max_retries=tool_config.max_retries,
                timeout=tool_config.timeout,
            )
            return create_http_tool(name, description, argument_contract, http_config)

        # 関数が渡された場合、通常のツールを作成
        func = func_or_url
        func.__doc__ = func.__doc__ or description

        @tool(name, args_schema=argument_contract)
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                retries = tool_config.max_retries
                while retries > 0:
                    try:
                        return await func(*args, **kwargs)
                    except Exception as e:
                        if retries <= 1 or not tool_config.handle_errors:
                            raise
                        retries -= 1
                        await asyncio.sleep(1)
            except Exception as e:
                if tool_config.handle_errors:
                    return f'Error executing tool: {str(e)}, please retry with fix'
                raise

        @tool(name, args_schema=argument_contract)
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                retries = tool_config.max_retries
                while retries > 0:
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        if retries <= 1 or not tool_config.handle_errors:
                            raise
                        retries -= 1
                        asyncio.sleep(1)
            except Exception as e:
                if tool_config.handle_errors:
                    return f'Error executing tool: {str(e)}, please retry with fix'
                raise

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator

def create_http_tool(
    name: str,
    description: Optional[str],
    argument_contract: Optional[Type[BaseModel]],
    config: HTTPToolConfig,
) -> BaseTool:
    """HTTP FloTool の作成

    Args:
        name (str): ツールの名前
        description (Optional[str]): ツールの説明
        argument_contract (Optional[Type[BaseModel]]): 引数のスキーマ
        config (HTTPToolConfig): HTTPツールの設定

    Returns:
        BaseTool: HTTP FloTool
    """
    class HTTPTool(BaseTool):
        name = name
        description = description or f"HTTP tool for {config.base_url}"
        args_schema = argument_contract
        
        def __init__(self):
            super().__init__()
            self.client = httpx.AsyncClient(
                base_url=config.base_url,
                headers=config.headers,
                timeout=config.timeout
            )

        async def _arun(self, *args: Any, **kwargs: Any) -> Any:
            try:
                retries = config.max_retries
                while retries > 0:
                    try:
                        response = await self.client.post(
                            "/",
                            json={"args": args, "kwargs": kwargs}
                        )
                        response.raise_for_status()
                        return response.json()
                    except Exception as e:
                        if retries <= 1 or not config.handle_errors:
                            raise
                        retries -= 1
                        await asyncio.sleep(1)
            except Exception as e:
                if config.handle_errors:
                    return f'Error executing HTTP tool: {str(e)}, please retry'
                raise

        def _run(self, *args: Any, **kwargs: Any) -> Any:
            return asyncio.run(self._arun(*args, **kwargs))

    return HTTPTool()