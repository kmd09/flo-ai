# flo_ai/router/flo_router.py
import functools
from typing import Union, Optional
from abc import ABC, abstractmethod
from uuid import UUID
from langgraph.graph import END, StateGraph
from flo_ai.models.flo_node import FloNode
from flo_ai.state.flo_session import FloSession
from flo_ai.models.flo_team import FloTeam
from flo_ai.models.flo_member import FloMember
from flo_ai.models.flo_routed_team import FloRoutedTeam
from flo_ai.constants.prompt_constants import FLO_FINISH
from flo_ai.models.flo_executable import ExecutableType
from flo_ai.state.flo_state import (
    TeamFloAgentState,
    STATE_NAME_LOOP_CONTROLLER,
    STATE_NAME_NEXT,
)
from flo_ai.constants.flo_node_contants import (
    INTERNAL_NODE_REFLECTION_MANAGER,
    INTERNAL_NODE_DELEGATION_MANAGER,
)
from flo_ai.types.http import (
    RouterType,
    HTTPRouterConfig,
)

class FloRouter(ABC):
    """Router base class"""
    def __init__(
        self,
        session: FloSession,
        name: str,
        flo_team: FloTeam,
        executor,
        model_name: Union[str, None] = 'default',
    ):
        self.name = name
        self.session: FloSession = session
        self.flo_team: FloTeam = flo_team
        self.members = flo_team.members
        self.member_names = [x.name for x in flo_team.members]
        self.type: ExecutableType = ExecutableType.router
        self.executor = executor
        self.model_name = model_name

    def build_routed_team(self) -> FloRoutedTeam:
        """Build the routed team"""
        return self.build_graph()

    @abstractmethod
    def build_graph(self):
        """Build the routing graph"""
        pass

    def build_node(self, flo_member: FloMember) -> FloNode:
        """Build a node from a FloMember"""
        node_builder = FloNode.Builder(self.session)
        if flo_member.type == ExecutableType.router:
            return node_builder.build_from_router(flo_member)
        if flo_member.type == ExecutableType.team:
            return node_builder.build_from_team(flo_member)
        if flo_member.type == ExecutableType.delegator:
            return node_builder.build_from_delegator(flo_member)
        if flo_member.type == ExecutableType.reflection:
            return node_builder.build_from_reflection(flo_member)
        return node_builder.build_from_agent(flo_member)

    def router_fn(self, state: TeamFloAgentState):
        """Default routing function"""
        next = state['next']
        conditional_map = {k: k for k in self.member_names}
        conditional_map[FLO_FINISH] = END
        self.session.append(node=next)
        if self.session.is_looping(node=next):
            return conditional_map[FLO_FINISH]
        return conditional_map[next]

    def update_reflection_state(
        self, state: TeamFloAgentState, reflection_agent_name: str
    ):
        """Update reflection state"""
        tracker = state.get(STATE_NAME_LOOP_CONTROLLER) or {}
        tracker[reflection_agent_name] = tracker.get(reflection_agent_name, 0) + 1
        return {STATE_NAME_LOOP_CONTROLLER: tracker}

    def add_delegation_edge(
        self,
        workflow: StateGraph,
        parent: FloNode,
        delegation_node: FloNode,
        nextNode: Union[FloNode, str],
    ):
        """Add delegation edge to the graph"""
        to_agent_names = delegation_node.delegate.to
        delegation_node_name = delegation_node.name
        next_node_name = nextNode if isinstance(nextNode, str) else nextNode.name

        retry = delegation_node.delegate.retry or 1

        conditional_map = {agent_name: agent_name for agent_name in to_agent_names}
        conditional_map[next_node_name] = next_node_name

        workflow.add_node(
            INTERNAL_NODE_DELEGATION_MANAGER,
            functools.partial(
                self.update_reflection_state, reflection_agent_name=delegation_node_name
            ),
        )

        workflow.add_edge(parent.name, INTERNAL_NODE_DELEGATION_MANAGER)
        workflow.add_conditional_edges(
            INTERNAL_NODE_DELEGATION_MANAGER,
            self.__get_refelection_routing_fn(
                retry, delegation_node_name, next_node_name
            ),
            {
                delegation_node_name: delegation_node_name,
                next_node_name: next_node_name,
            },
        )

        workflow.add_conditional_edges(
            delegation_node_name,
            FloRouter.__get_delegation_router_fn(next_node_name),
            conditional_map,
        )

    def add_reflection_edge(
        self,
        workflow: StateGraph,
        reflection_node: FloNode,
        nextNode: Union[FloNode, str],
    ):
        """Add reflection edge to the graph"""
        to_agent_name = reflection_node.delegate.to[0]
        retry = reflection_node.delegate.retry or 1
        reflection_agent_name = reflection_node.name
        next = nextNode if isinstance(nextNode, str) else nextNode.name

        workflow.add_node(
            INTERNAL_NODE_REFLECTION_MANAGER,
            functools.partial(
                self.update_reflection_state,
                reflection_agent_name=reflection_agent_name,
            ),
        )

        workflow.add_edge(to_agent_name, INTERNAL_NODE_REFLECTION_MANAGER)
        workflow.add_conditional_edges(
            INTERNAL_NODE_REFLECTION_MANAGER,
            self.__get_refelection_routing_fn(retry, reflection_agent_name, next),
            {reflection_agent_name: reflection_agent_name, next: next},
        )
        workflow.add_edge(reflection_agent_name, to_agent_name)

    @staticmethod
    def __get_reflection_routing_fn(
        retries: int, reflection_agent_name, next_node_name
    ):
        def reflection_routing_fn(state: TeamFloAgentState):
            tracker = state[STATE_NAME_LOOP_CONTROLLER]
            if (
                tracker is not None
                and reflection_agent_name in tracker
                and tracker[reflection_agent_name] > retries
            ):
                return next_node_name
            return reflection_agent_name

        return reflection_routing_fn

    @staticmethod
    def __get_delegation_router_fn(nextNode: str):
        def delegation_router(state: TeamFloAgentState):
            if STATE_NAME_NEXT not in state:
                return nextNode
            return state[STATE_NAME_NEXT]

        return delegation_router

    @classmethod
    def create(
        cls,
        session: FloSession,
        name: str,
        team: FloTeam,
        router_config: Optional[Union[dict, HTTPRouterConfig]] = None,
    ):
        """Factory method to create router"""
        if router_config is not None:
            # If HTTPRouterConfig is provided, delegate to HTTP router
            from flo_ai.router.http.router import create_http_router
            if isinstance(router_config, dict):
                router_config = HTTPRouterConfig(**router_config)
            return create_http_router(
                router_type=RouterType(router_config.router_type),
                config=router_config,
                session_id=session.session_id
            )
        
        # Default local router creation
        return cls.Builder(
            session=session,
            name=name,
            flo_team=team,
        ).build()
