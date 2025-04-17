import streamlit as st
from text2sql import AppState
from text2sql.openai_utils import (
    tracked_chat_completion,
)
from text2sql.components import model_selector

st.title("간단한 챗봇")

# State 초기화
state = AppState(st.session_state)

# 모델 선택기 컴포넌트 사용
model_selector(state)

# 구분선
st.divider()

# 기존 채팅 기록 표시
for message in state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력
if prompt := st.chat_input("메시지를 입력하세요"):
    # 사용자 메시지 추가
    state.add_message("user", prompt)

    # 화면에 즉시 표시
    with st.chat_message("user"):
        st.markdown(prompt)

    # AI 응답 표시
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        # 추적 기능이 포함된 함수 호출
        stream = tracked_chat_completion(
            model=state.selected_model,
            messages=state.messages,
            stream=True,
        )

        # 응답 처리
        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                full_response += chunk.choices[0].delta.content
                message_placeholder.markdown(full_response + "▌")

        # 최종 응답 표시
        message_placeholder.markdown(full_response)

        # 응답 저장
        state.add_message("assistant", full_response)
