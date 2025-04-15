import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
import os

from text2sql.graphs.simple_llm_graph import simple_llm_graph
from text2sql.state import AppState

st.title("LLM을 사용한 LangGraph 예제")

openai_api_key = st.text_input(
    "OpenAI API 키를 입력하세요",
    type="password",
    value=st.secrets.get("OPENAI_API_KEY", ""),
)
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key


if "messages" not in st.session_state:
    st.session_state.messages = []

state = AppState(st.session_state)


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text
        self.placeholder = self.container.empty()

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.placeholder.markdown(self.text)


for message in state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("메시지를 입력하세요"):
    if not openai_api_key:
        st.error("OpenAI API 키를 입력해주세요.")
    else:
        # 사용자 메시지 추가
        state.add_message("user", prompt)

        with st.chat_message("user"):
            st.markdown(prompt)

        # AI 응답 표시
        with st.chat_message("assistant"):
            response_placeholder = st.empty()

            with st.spinner("응답 생성 중..."):
                result = simple_llm_graph.invoke({"question": prompt, "answer": ""})

            response_placeholder.markdown(result["answer"])

            state.add_message("assistant", result["answer"])
