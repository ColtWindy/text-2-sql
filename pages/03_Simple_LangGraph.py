import streamlit as st
from text2sql.graphs.mock_graph import mock_graph

st.title("간단한 LangGraph 예제")


def simple_langgraph_example():
    # 이미 컴파일된 그래프 사용
    result = mock_graph.invoke({"message": "시작 데이터"})
    return result["message"]


if st.button("LangGraph 실행"):
    result = simple_langgraph_example()
    st.write("결과:", result)
