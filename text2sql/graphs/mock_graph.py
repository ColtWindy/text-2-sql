# 상태 타입 정의
from typing import TypedDict
from langgraph.graph import StateGraph
from typing import TypedDict

class State(TypedDict):
    message: str

# 그래프 생성 및 컴파일 (전역 변수로 선언)
def create_graph():
    # 그래프 생성 (state_schema 제공)
    graph = StateGraph(state_schema=State)

    # 노드 추가
    def node1(state: State) -> State:
        return {"message": state["message"] + " 노드1에서 처리됨"}

    def node2(state: State) -> State:
        return {"message": state["message"] + " 노드2에서 처리됨"}

    # 노드를 그래프에 추가
    graph.add_node("node1", node1)
    graph.add_node("node2", node2)

    # 엣지 추가
    graph.add_edge("node1", "node2")

    # 시작점 설정
    graph.set_entry_point("node1")

    # 그래프 컴파일
    return graph.compile()

# 전역 변수로 컴파일된 그래프 생성
mock_graph = create_graph()
