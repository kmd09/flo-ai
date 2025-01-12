# flo_ai/router/http/router.py
import asyncio
import httpx
from typing import List, Optional
from uuid import UUID

from flo_ai.state.flo_state import TeamFloAgentState
from flo_ai.constants.prompt_constants import FLO_FINISH
from flo_ai.types.http import (
    RouterType,
    HTTPRouterConfig,
    HTTPRouterRequest,
    HTTPRouterResponse,
    RouterState,
    HTTPMetadata,
    HTTPMessage
)
from flo_ai.error.http.exception import (
    HTTPFloError,
    HTTPTimeoutError,
    HTTPRetryExhaustedError,
    HTTPRouterError
)

class HTTPFloRouter:
    """HTTP based FloRouter implementation"""
    def __init__(
        self, 
        config: HTTPRouterConfig,
        session_id: Optional[UUID] = None
    ):
        self.config = config
        self.client = httpx.AsyncClient(
            base_url=str(config.base_url),
            timeout=config.timeout,
            headers=config.headers
        )
        self.session_id = session_id

    async def route(
        self, 
        state: TeamFloAgentState,
        available_nodes: List[str]
    ) -> HTTPRouterResponse:
        """Route the request to the appropriate node"""
        metadata = HTTPMetadata(session_id=self.session_id)
        
        router_state = RouterState(
            messages=[HTTPMessage(**msg.dict()) for msg in state.get("messages", [])],
            current_node=state.get("next", ""),
            loop_tracker=state.get("loop_tracker", {}),
            metadata=metadata
        )

        request = HTTPRouterRequest(
            state=router_state,
            available_nodes=available_nodes + [FLO_FINISH],
            metadata=metadata
        )

        return await self._make_request(request)

class HTTPFloRouter:
    async def _make_request(self, request: HTTPRouterRequest) -> HTTPRouterResponse:
        retries = self.config.retry_count
        
        while retries > 0:
            try:
                response = await self.client.post(
                    "/route",
                    json=request.dict(exclude_none=True)
                )
                response.raise_for_status()
                return HTTPRouterResponse(**response.json())
            
            except httpx.TimeoutException as e:
                if retries <= 1:
                    raise HTTPTimeoutError(details={"original_error": str(e)})
                retries -= 1
                await asyncio.sleep(self.config.retry_delay)
            
            except httpx.HTTPError as e:
                if retries <= 1:
                    raise HTTPRouterError(
                        router_name=self.name,
                        message=f"HTTP request failed: {str(e)}",
                        details={
                            "status_code": e.response.status_code if hasattr(e, 'response') else None,
                            "original_error": str(e)
                        }
                    )
                retries -= 1
                await asyncio.sleep(self.config.retry_delay)
            
            except Exception as e:
                raise HTTPRouterError(
                    router_name=self.name,
                    message=f"Unexpected error: {str(e)}",
                    details={"original_error": str(e)}
                )

        raise HTTPRetryExhaustedError(
            details={
                "router_name": self.name,
                "max_retries": self.config.retry_count
            }
        )

    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()

class HTTPFloSupervisorRouter(HTTPFloRouter):
    """Supervisor specific router implementation"""
    
    async def _make_request(self, request: HTTPRouterRequest) -> HTTPRouterResponse:
        """Add supervisor specific headers"""
        request.metadata.additional_kwargs["router_type"] = "supervisor"
        return await super()._make_request(request)

class HTTPFloLinearRouter(HTTPFloRouter):
    """Linear router implementation"""
    
    async def route(
        self, 
        state: TeamFloAgentState,
        available_nodes: List[str]
    ) -> HTTPRouterResponse:
        """Linear routing logic"""
        metadata = HTTPMetadata(session_id=self.session_id)
        current_node = state.get("next", available_nodes[0])
        
        try:
            current_idx = available_nodes.index(current_node)
            next_idx = current_idx + 1
            
            if next_idx >= len(available_nodes):
                return HTTPRouterResponse(
                    next_node=FLO_FINISH,
                    metadata=metadata
                )
            
            return HTTPRouterResponse(
                next_node=available_nodes[next_idx],
                metadata=metadata
            )
        except ValueError:
            return HTTPRouterResponse(
                next_node="ERROR",
                error=f"Node {current_node} not found in available nodes",
                metadata=metadata
            )

class HTTPFloLLMRouter(HTTPFloRouter):
    """LLM based router implementation"""
    
    async def _make_request(self, request: HTTPRouterRequest) -> HTTPRouterResponse:
        """Add LLM specific headers"""
        request.metadata.additional_kwargs["router_type"] = "llm"
        request.metadata.additional_kwargs["model"] = self.config.headers.get("model", "default")
        return await super()._make_request(request)

class HTTPFloRouterFactory:
    """Factory for creating HTTP routers"""
    
    @staticmethod
    def create_router(
        config: HTTPRouterConfig,
        session_id: Optional[UUID] = None
    ) -> HTTPFloRouter:
        """Create appropriate router based on type"""
        router_map = {
            RouterType.SUPERVISOR: HTTPFloSupervisorRouter,
            RouterType.LINEAR: HTTPFloLinearRouter,
            RouterType.LLM: HTTPFloLLMRouter
        }
        
        router_class = router_map.get(config.router_type, HTTPFloRouter)
        return router_class(config, session_id)

def create_http_router(
    router_type: RouterType,
    config: HTTPRouterConfig,
    session_id: Optional[UUID] = None
) -> HTTPFloRouter:
    """Factory function to create router"""
    return HTTPFloRouterFactory.create_router(config, session_id)
