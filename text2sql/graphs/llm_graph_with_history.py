from typing import TypedDict, List, Union
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage, BaseMessage


# 상태 타입 정의 - 대화 히스토리 추가
class State(TypedDict):
    question: str
    answer: str
    history: List[dict]


# 노드 이름 상수 정의
class Node:
    GENERATE_ANSWER = "generate_answer"
    UPDATE_HISTORY = "update_history"


def create_graph():
    # LLM 모델 초기화
    llm = ChatOpenAI()

    # 그래프 생성
    graph = StateGraph(state_schema=State)

    # NODE: 히스토리를 고려한 답변 생성
    def generate_answer(state: State) -> State:
        # 시스템 메시지 생성
        system_message = SystemMessage(
            content="너는 친절하고, 정확한 질문에 답변해 주는 AI 비서야."
        )

        # 대화 히스토리 구성 - List[BaseMessage] 타입으로 명시
        messages: List[BaseMessage] = [system_message]

        # 과거 대화 내역 추가
        for message in state["history"]:
            if message["role"] == "user":
                messages.append(HumanMessage(content=message["content"]))
            elif message["role"] == "assistant":
                messages.append(AIMessage(content=message["content"]))

        # 현재 질문 추가
        messages.append(HumanMessage(content=state["question"]))

        # LLM에 요청
        response = llm.invoke(messages)
        content = response.content

        # LLM 응답이 리스트 형태로 반환되는 경우 문자열로 변환
        # 일부 모델은 특정 상황에서 응답을 리스트 형태로 반환할 수 있음
        # 이런 경우 문자열로 변환하여 일관된 형식 유지
        if isinstance(content, list):
            content = str(content)

        return {
            "question": state["question"],
            "answer": content,
            "history": state["history"],
        }

    # NODE: 히스토리 업데이트
    def update_history(state: State) -> State:
        # 현재 대화를 히스토리에 추가
        updated_history = state["history"].copy()
        updated_history.append({"role": "user", "content": state["question"]})
        updated_history.append({"role": "assistant", "content": state["answer"]})

        # 히스토리 길이 제한 (최근 10개 대화만 유지)
        if len(updated_history) > 20:
            updated_history = updated_history[-20:]

        return {
            "question": state["question"],
            "answer": state["answer"],
            "history": updated_history,
        }

    # 노드를 그래프에 추가
    graph.add_node(Node.GENERATE_ANSWER, generate_answer)
    graph.add_node(Node.UPDATE_HISTORY, update_history)

    # EDGE: 그래프 엣지 연결
    graph.add_edge(Node.GENERATE_ANSWER, Node.UPDATE_HISTORY)

    # ENTRY_POINT: 시작점 설정
    graph.set_entry_point(Node.GENERATE_ANSWER)

    # 그래프 컴파일
    return graph.compile()


llm_graph_with_history = create_graph()
