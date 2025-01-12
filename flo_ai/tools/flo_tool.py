# flo_ai/tools/flo_tool.py
import asyncio
from typing import Optional, Any, Type, Union, Dict, Callable
from langchain.tools import tool, BaseTool
from functools import wraps
from flo_ai.tools.http import HTTPToolFactory, HTTPRequestConfig
from pydantic import BaseModel

class FloToolConfig(BaseModel):
    """FloTool の設定"""
    handle_errors: bool = True  # エラーハンドリングを有効にするかどうか
    max_retries: int = 3       # リトライ回数
    timeout: int = 30          # タイムアウト時間（秒）

def flotool(
    name: str,
    description: Optional[str] = None,
    argument_contract: Optional[type] = None,
    config: Optional[Union[FloToolConfig, HTTPRequestConfig]] = None,
):
    """FloTool デコレータ

    Args:
        name (str): ツールの名前
        description (Optional[str], optional): ツールの説明
        argument_contract (Optional[type], optional): 引数のスキーマ
        config (Optional[HTTPRequestConfig]], optional): ツールの設定
    """
    def decorator(func_or_url: Union[Callable, str]):
        # デフォルト設定
        tool_config = config or FloToolConfig()

        # HTTP URLが渡された場合、HTTPツールを作成
        if isinstance(func_or_url, str):
            return HTTPToolFactory.create_tool(
                endpoint=func_or_url, 
                name=name, 
                description=description, 
                config=config)

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
