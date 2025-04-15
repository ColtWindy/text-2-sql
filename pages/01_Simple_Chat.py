import streamlit as st
from text2sql import AppState
from text2sql.openai_utils import (
    tracked_chat_completion,
    initialize_models,
    refresh_models,
)

st.title("간단한 챗봇")

# State 초기화
state = AppState(st.session_state)

# 모델 초기화
initialize_models(state)

# 모델 정보 fold
with st.expander("현재 설정"):
    st.write(f"현재 모델: {state.selected_model}")
    st.write("모델을 변경하려면 아래 드롭다운을 사용하세요.")

    # 모델 목록 새로고침 버튼
    if st.button("모델 목록 새로고침"):
        if refresh_models(state):
            st.success("모델 목록이 업데이트되었습니다!")
            st.rerun()  # 페이지 새로고침
        else:
            st.error("모델 목록 업데이트에 실패했습니다.")

selected_index = state.available_models.index(state.selected_model)
selected_model = st.selectbox("모델:", state.available_models, index=selected_index)

# 선택한 모델 저장
state.selected_model = selected_model

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
