import streamlit as st
from text2sql import AppState
from text2sql.openai_utils import (
    tracked_chat_completion,
    initialize_models,
)
from text2sql.components import model_selector

st.title("간단한 SQL 생성기")
state = AppState(st.session_state)
initialize_models(state)

SQL_SYSTEM_PROMPT = """당신은 전문 SQL 생성기입니다. 사용자는 한국어로 질문할 것이며, 당신은 유효한 PostgreSQL SQL 쿼리를 출력해야 합니다. 주어진 스키마 정보를 사용하고 스키마에 없는 테이블/컬럼은 가정하지 마세요. 질문이 모호하거나 세부 정보가 누락된 경우, 추측하지 말고 명확한 질문을 해주세요."""

DB_SCHEMA = """
테이블: users
- id (정수, 기본 키)
- username (문자열)
- email (문자열)
- created_at (타임스탬프)

테이블: orders
- id (정수, 기본 키)
- user_id (정수, 외래 키 -> users.id)
- total_amount (실수)
- status (문자열)
- created_at (타임스탬프)

테이블: products
- id (정수, 기본 키)
- name (문자열)
- price (실수)
- category (문자열)
- stock (정수)
"""

with st.sidebar:
    with st.expander("데이터베이스 스키마 정보"):
        st.code(DB_SCHEMA)

# 모델 선택기 컴포넌트 사용
model_selector(state)

st.divider()

if not state.messages:
    full_system_prompt = f"{SQL_SYSTEM_PROMPT}\n\n스키마 정보:\n{DB_SCHEMA}"
    state.messages = [{"role": "system", "content": full_system_prompt}]

# 기존 채팅 기록 표시 (시스템 메시지는 표시하지 않음)
for message in state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# 사용자 입력
if prompt := st.chat_input("SQL로 변환할 질문을 입력하세요"):
    state.add_message("user", prompt)

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        # langgchain 래퍼 호출
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

        message_placeholder.markdown(full_response)

        state.add_message("assistant", full_response)
