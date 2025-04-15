import streamlit as st
import os

from text2sql.graphs.llm_graph_with_history import llm_graph_with_history
from text2sql.state import AppState

# 페이지 설정
st.title("대화 기록이 있는 챗봇")

# OpenAI API 키 입력 (보안을 위해 st.secrets 사용 권장)
openai_api_key = st.text_input(
    "OpenAI API 키를 입력하세요",
    type="password",
    value=st.secrets.get("OPENAI_API_KEY", ""),
)
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key

# 세션 상태 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

state = AppState(st.session_state)

# 이전 메시지 표시
for message in state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력
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
            message_placeholder = st.empty()

            with st.spinner("대화 내역을 분석하고 응답을 생성 중..."):
                # 그래프 호출
                result = llm_graph_with_history.invoke(
                    {
                        "question": prompt,
                        "answer": "",
                        "history": [
                            {"role": msg["role"], "content": msg["content"]}
                            for msg in state.messages[
                                :-1
                            ]  # 마지막 사용자 메시지 제외 (이미 question에 포함)
                        ],
                    }
                )

            # 결과 표시
            answer = result["answer"]
            message_placeholder.markdown(answer)

            # 응답 저장
            state.add_message("assistant", answer)

            # 히스토리 업데이트
            st.session_state.chat_history = result["history"]
