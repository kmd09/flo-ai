import functools
from flo_ai.models.flo_agent import FloAgent
from flo_ai.models.flo_reflection_agent import FloReflectionAgent
from flo_ai.models.flo_routed_team import FloRoutedTeam
from flo_ai.models.delegate import Delegate
from langchain.agents import AgentExecutor
from flo_ai.state.flo_state import TeamFloAgentState, STATE_NAME_MESSAGES
from flo_ai.models.flo_member import FloMember
from langchain_core.messages import AIMessage, HumanMessage
from flo_ai.models.flo_executable import ExecutableType
from flo_ai.state.flo_session import FloSession
from typing import Optional, Type, List
from flo_ai.callbacks.flo_callbacks import (
    FloAgentCallback,
    FloRouterCallback,
    FloCallback,
)
from flo_ai.common.flo_logger import get_logger
from flo_ai.state.flo_output_collector import FloOutputCollector
from abc import ABC
from typing import Any, Dict
from langchain_core.runnables import Runnable

class FloNode(FloMember, ABC):
    def __init__(
        self, 
        name: str, 
        type: str = "node",
        model_name: Optional[str] = None,
        executor: Optional[Runnable] = None
    ) -> None:
        """
        Args:
            name: ノードの名前
            type: ノードの種類
            model_name: 使用するモデル名
            executor: 実行エンジン（ローカル実行用）
        """
        super().__init__(name, type)
        self.model_name = model_name
        self.executor = executor

    async def ainvoke(
        self, 
        state: Dict[str, Any], 
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """ノードの非同期実行

        Args:
            state: 現在の状態
            config: 実行時設定

        Returns:
            Dict[str, Any]: 更新された状態

        Raises:
            NotImplementedError: 未実装の場合
        """
        raise NotImplementedError("ainvoke must be implemented by subclass")

    def invoke(
        self, 
        state: Dict[str, Any], 
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """ノードの同期実行

        Args:
            state: 現在の状態
            config: 実行時設定

        Returns:
            Dict[str, Any]: 更新された状態
        """
        if self.executor is None:
            raise ValueError("Executor is required for synchronous invocation")
        return self.executor.invoke(state, config)

    def draw(self, xray: bool = True) -> Optional[bytes]:
        """ノードの可視化

        Args:
            xray: 詳細表示フラグ

        Returns:
            Optional[bytes]: 可視化結果
        """
        if self.executor and hasattr(self.executor, 'get_graph'):
            return self.executor.get_graph().draw_mermaid_png()
        return None
    
    @staticmethod
    def _get_last_message(state: Dict[str, Any]) -> str:
        """最後のメッセージを取得"""
        return state[STATE_NAME_MESSAGES][-1].content

    @staticmethod
    def _join_graph(response: dict):
        """グラフの結果を結合"""
        return {STATE_NAME_MESSAGES: [response[STATE_NAME_MESSAGES][-1]]}

    @staticmethod
    def _filter_callbacks(session, type: Type) -> List:
        """指定された型のコールバックをフィルタリング"""
        cbs = session.callbacks
        return list(filter(lambda callback: isinstance(callback, type), cbs))

    class Builder:
        def __init__(self, session: FloSession) -> None:
            self.session = session

        def build_from_agent(self, flo_agent: FloAgent) -> 'FloNode':
            agent_func = functools.partial(
                FloNode.Builder.__teamflo_agent_node,
                agent=flo_agent.runnable,
                name=flo_agent.name,
                session=self.session,
                model_name=flo_agent.model_name,
                data_collector=flo_agent.data_collector,
            )
            agent_func_async = functools.partial(
                FloNode.Builder.__async_teamflo_agent_node,
                agent=flo_agent.runnable,
                name=flo_agent.name,
                session=self.session,
                model_name=flo_agent.model_name,
                data_collector=flo_agent.data_collector,
            )
            return FloNode(
                agent_func,
                flo_agent.name,
                flo_agent.type,
                async_func=agent_func_async,
                agent_executable=flo_agent.runnable,
            )

        def build_from_reflection(self, flo_agent: FloReflectionAgent) -> 'FloNode':
            agent_func = functools.partial(
                FloNode.Builder.__teamflo_agent_node,
                agent=flo_agent.runnable,
                name=flo_agent.name,
                session=self.session,
                model_name=flo_agent.model_name,
            )
            return FloNode(
                agent_func,
                flo_agent.name,
                flo_agent.type,
                delegate=flo_agent.delegate,
                agent_executable=flo_agent.runnable,
            )

        def build_from_team(self, flo_team: FloRoutedTeam) -> 'FloNode':
            team_chain = (
                functools.partial(
                    FloNode.Builder.__teamflo_team_node, members=flo_team.runnable.nodes
                )
                | flo_team.runnable
            )
            return FloNode(
                (
                    FloNode.Builder.__get_last_message
                    | team_chain
                    | FloNode.Builder.__join_graph
                ),
                flo_team.name,
                flo_team.type,
                agent_executable=flo_team.runnable,
            )

        def build_from_router(self, flo_router) -> 'FloNode':
            router_func = functools.partial(
                FloNode.Builder.__teamflo_router_node,
                agent=flo_router.executor,
                name=flo_router.name,
                session=self.session,
                model_name=flo_router.model_name,
            )
            return FloNode(
                router_func,
                flo_router.name,
                flo_router.type,
                agent_executable=flo_router.executor,
            )

        def build_from_delegator(self, flo_router) -> 'FloNode':
            router_func = functools.partial(
                FloNode.Builder.__teamflo_router_node,
                agent=flo_router.executor,
                name=flo_router.name,
                session=self.session,
                model_name=flo_router.model_name,
            )
            return FloNode(
                router_func,
                flo_router.name,
                flo_router.type,
                delegate=flo_router.delegate,
                agent_executable=flo_router.executor,
            )

        @staticmethod
        def __teamflo_agent_node(
            state: TeamFloAgentState,
            agent: AgentExecutor,
            name: str,
            session: FloSession,
            model_name: str,
            data_collector: Optional[FloOutputCollector] = None,
        ):
            agent_cbs: List[FloAgentCallback] = FloNode.Builder.__filter_callbacks(
                session, FloAgentCallback
            )
            flo_cbs: List[FloCallback] = FloNode.Builder.__filter_callbacks(
                session, FloCallback
            )
            [
                callback.on_agent_start(name, model_name, state['messages'], **{})
                for callback in agent_cbs
            ]
            [
                callback.on_agent_start(name, model_name, state['messages'], **{})
                for callback in flo_cbs
            ]
            try:
                result = agent.invoke(state)
                output = result if isinstance(result, str) else result['output']
                if data_collector is not None:
                    get_logger().info(
                        'appending output to data collector', session=session
                    )
                    data_collector.append(output)
            except Exception as e:
                [
                    callback.on_agent_error(name, model_name, e, **{})
                    for callback in agent_cbs
                ]
                [
                    callback.on_agent_error(name, model_name, e, **{})
                    for callback in flo_cbs
                ]
                raise e
            [
                callback.on_agent_end(name, model_name, output, **{})
                for callback in agent_cbs
            ]
            [
                callback.on_agent_start(name, model_name, output, **{})
                for callback in flo_cbs
            ]
            return {STATE_NAME_MESSAGES: [AIMessage(content=output, name=name)]}

        @staticmethod
        async def __async_teamflo_agent_node(
            state: TeamFloAgentState,
            agent: AgentExecutor,
            name: str,
            session: FloSession,
            model_name: str,
            data_collector: Optional[FloOutputCollector] = None,
        ):
            agent_cbs: List[FloAgentCallback] = FloNode.Builder.__filter_callbacks(
                session, FloAgentCallback
            )
            flo_cbs: List[FloCallback] = FloNode.Builder.__filter_callbacks(
                session, FloCallback
            )
            [
                callback.on_agent_start(name, model_name, state['messages'], **{})
                for callback in agent_cbs
            ]
            [
                callback.on_agent_start(name, model_name, state['messages'], **{})
                for callback in flo_cbs
            ]
            try:
                result = await agent.ainvoke(state)
                output = result if isinstance(result, str) else result['output']
                if data_collector is not None:
                    get_logger().info(
                        'appending output to data collector', session=session
                    )
                    data_collector.append(output)
            except Exception as e:
                [
                    callback.on_agent_error(name, model_name, e, **{})
                    for callback in agent_cbs
                ]
                [
                    callback.on_agent_error(name, model_name, e, **{})
                    for callback in flo_cbs
                ]
                raise e
            [
                callback.on_agent_end(name, model_name, output, **{})
                for callback in agent_cbs
            ]
            [
                callback.on_agent_start(name, model_name, output, **{})
                for callback in flo_cbs
            ]
            return {STATE_NAME_MESSAGES: [AIMessage(content=output, name=name)]}

        @staticmethod
        def __filter_callbacks(session: FloSession, type: Type):
            cbs = session.callbacks
            return list(filter(lambda callback: isinstance(callback, type), cbs))

        @staticmethod
        def __teamflo_router_node(
            state: TeamFloAgentState,
            agent: AgentExecutor,
            name: str,
            session: FloSession,
            model_name: str,
        ):
            agent_cbs: List[FloRouterCallback] = FloNode.Builder.__filter_callbacks(
                session, FloRouterCallback
            )
            flo_cbs: List[FloCallback] = FloNode.Builder.__filter_callbacks(
                session, FloCallback
            )
            [
                callback.on_router_start(name, model_name, state['messages'], **{})
                for callback in agent_cbs
            ]
            [
                callback.on_router_start(name, model_name, state['messages'], **{})
                for callback in flo_cbs
            ]
            try:
                result = agent.invoke(state)
                nextNode = result if isinstance(result, str) else result['next']
            except Exception as e:
                [
                    callback.on_router_error(name, model_name, e, **{})
                    for callback in agent_cbs
                ]
                [
                    callback.on_router_error(name, model_name, e, **{})
                    for callback in flo_cbs
                ]
                raise e
            [
                callback.on_router_end(name, model_name, nextNode, **{})
                for callback in agent_cbs
            ]
            [
                callback.on_router_start(name, model_name, nextNode, **{})
                for callback in flo_cbs
            ]
            return {'next': nextNode}

        @staticmethod
        def __get_last_message(state: TeamFloAgentState) -> str:
            return state[STATE_NAME_MESSAGES][-1].content

        @staticmethod
        def __join_graph(response: dict):
            return {STATE_NAME_MESSAGES: [response[STATE_NAME_MESSAGES][-1]]}

        @staticmethod
        def __teamflo_team_node(message: str, members: list[str]):
            results = {
                STATE_NAME_MESSAGES: [HumanMessage(content=message)],
                'team_members': ', '.join(members),
            }
            return results

class LocalFloNode(FloNode):
    """ローカル実行用のノード実装"""
    
    def __init__(
        self, 
        name: str,
        executor: Runnable,
        model_name: Optional[str] = None,
        type: str = "local",
        session = None
    ) -> None:
        super().__init__(
            name=name,
            type=type,
            model_name=model_name,
            executor=executor
        )
        self.session = session

    async def ainvoke(
        self, 
        state: Dict[str, Any], 
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """ローカルノードの非同期実行（コールバック含む）"""
        agent_cbs: List[FloAgentCallback] = self._filter_callbacks(
            self.session, 
            FloAgentCallback
        )
        flo_cbs: List[FloCallback] = self._filter_callbacks(
            self.session, 
            FloCallback
        )

        # コールバックの実行
        [callback.on_agent_start(self.name, self.model_name, state['messages']) 
         for callback in agent_cbs + flo_cbs]

        try:
            result = await self.executor.ainvoke(state, config)
            output = result if isinstance(result, str) else result['output']
        except Exception as e:
            [callback.on_agent_error(self.name, self.model_name, e) 
             for callback in agent_cbs + flo_cbs]
            raise e

        [callback.on_agent_end(self.name, self.model_name, output) 
         for callback in agent_cbs + flo_cbs]

        return {STATE_NAME_MESSAGES: [AIMessage(content=output, name=self.name)]}

class HTTPFloNode(FloNode):
    """HTTPベースのノード実装（プレースホルダー）"""
    pass

class Builder:
    """ノードビルダー"""
    def __init__(self, session) -> None:
        self.session = session

    def build_from_agent(self, agent) -> FloNode:
        """エージェントからノードを構築"""
        if getattr(agent, 'type', None) == "http":
            from flo_ai.models.http.node import HTTPFloNodeFactory
            return HTTPFloNodeFactory.create_node(
                endpoint=agent.endpoint,
                name=agent.name,
                model_name=agent.model_name
            )
        else:
            return LocalFloNode(
                name=agent.name,
                executor=agent.executor,
                model_name=agent.model_name,
                session=self.session
            )
        