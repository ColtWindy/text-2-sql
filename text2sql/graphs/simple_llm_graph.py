from typing import TypedDict
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage


# 노드 이름을 str과 Enum을 상속받아 정의
class Node:
    PROCESS_QUESTION = "process_question"
    GENERATE_ANSWER = "generate_answer"
    FORMAT_RESULT = "format_result"


# 상태 타입 정의
class State(TypedDict):
    question: str
    answer: str


def create_graph():
    # LLM 모델 초기화
    llm = ChatOpenAI()

    graph = StateGraph(state_schema=State)

    # NODE: 질문 처리
    def process_question(state: State) -> State:
        return {"question": state["question"], "answer": ""}

    # NODE: LLM을 사용한 답변 생성
    def generate_answer(state: State) -> State:
        messages = [HumanMessage(content=state["question"])]
        response = llm.invoke(messages)
        # 타입 오류 수정
        content = response.content
        if isinstance(content, list):
            content = str(content)
        return {"question": state["question"], "answer": content}

    # NODE: 결과 포매팅
    def format_result(state: State) -> State:
        return {"question": state["question"], "answer": f"**{state['answer']}**"}

    # 노드를 그래프에 추가
    graph.add_node(Node.PROCESS_QUESTION, process_question)
    graph.add_node(Node.GENERATE_ANSWER, generate_answer)
    graph.add_node(Node.FORMAT_RESULT, format_result)

    # EDGE:
    graph.add_edge(Node.PROCESS_QUESTION, Node.GENERATE_ANSWER)
    # EDGE:
    graph.add_edge(Node.GENERATE_ANSWER, Node.FORMAT_RESULT)

    # ENTRY_POINT:
    graph.set_entry_point(Node.PROCESS_QUESTION)

    # 그래프 컴파일
    return graph.compile()


simple_llm_graph = create_graph()
