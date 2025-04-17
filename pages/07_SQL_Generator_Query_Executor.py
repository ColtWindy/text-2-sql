import streamlit as st
import re

# 커스텀 모듈 임포트
from text2sql import AppState
from text2sql.openai_utils import (
    tracked_chat_completion,
    initialize_models,
)
from text2sql.components import (
    model_selector,
    read_file_content,
    context_file_selector,
    display_query_results,
)
from text2sql.db_utils import execute_query, get_all_tables, get_table_schema

# 페이지 설정
st.title("SQL 생성 및 실행기")
st.write("데이터 분석가와 오퍼레이터를 위한 자연어 기반 데이터베이스 조회 도구입니다.")

# 애플리케이션 상태 초기화
state = AppState(st.session_state)

# 모델 초기화 실행
initialize_models(state)

# SQL 시스템 프롬프트 설정
SQL_SYSTEM_PROMPT = """당신은 전문 SQL 생성기입니다. 
사용자는 한국어로 질문할 것이며, 당신은 유효한 PostgreSQL SQL 쿼리를 출력해야 합니다. 
주어진 스키마 정보를 사용하고 스키마에 없는 테이블/컬럼은 가정하지 마세요.

아래 규칙을 반드시 지켜주세요:
1. 마크다운 코드 블록(```sql)을 사용해서 원시 SQL 구문만 출력하세요.
2. 원시 SQL 구문만 출력하세요.
3. 서브 쿼리가 필요한 경우, 각각의 단계를 명확히 하여 sql 주석에 추가 해 주세요.

질문이 모호하거나 더 많은 정보가 필요하면 사용자에게 명확하게 해달라고 요청하세요."""


# 상태 초기화 함수
def initialize_state():
    """상태 변수를 초기화하는 함수"""
    if not state.has("selected_context_files"):
        state.set("selected_context_files", [])
    if not state.has("generated_sql"):
        state.set("generated_sql", "")
    if not state.has("sql_messages"):
        full_system_prompt = build_prompt_context()
        state.set("sql_messages", [{"role": "system", "content": full_system_prompt}])


# 스키마 정보 조회
def get_db_schema():
    schema_text = ""
    tables = get_all_tables()
    if tables:
        for table in tables:
            schema_text += f"\n테이블: {table}\n"

            columns = get_table_schema(table)
            if columns:
                for col in columns:
                    col_name = col[0]
                    data_type = col[1]
                    nullable = "NULL 허용" if col[2] == "YES" else "NOT NULL"
                    schema_text += f"- {col_name} ({data_type}, {nullable})\n"

            # TODO: PK, FK, Relation?? 처리 할 것

    return schema_text


# 시스템 메시지 업데이트
def update_system_message():
    """시스템 메시지를 최신 컨텍스트로 업데이트"""
    if state.has("sql_messages"):
        messages = state.get("sql_messages")
        if messages and messages[0]["role"] == "system":
            # 업데이트된 프롬프트로 교체
            full_system_prompt = build_prompt_context()
            messages[0]["content"] = full_system_prompt
            state.set("sql_messages", messages)
            return True
    return False


# SQL 응답 정제
def clean_sql_response(response):
    """AI 응답에서 SQL 코드만 추출"""
    cleaned_response = response.strip()

    # ```sql ... ``` 패턴 찾기 (정규식 사용)
    sql_pattern = re.compile(r"```sql\s*(.*?)\s*```", re.DOTALL)
    match = sql_pattern.search(cleaned_response)

    if match:
        sql = match.group(1).strip()
        if sql:
            return sql

    # SQL 구문을 찾지 못한 경우 빈 문자열 반환
    return ""


# 프롬프트 컨텍스트 생성 함수
def build_prompt_context():
    base_prompt = f"{SQL_SYSTEM_PROMPT}\n\n스키마 정보:\n{db_schema}"

    # 선택된 컨텍스트 파일이 있다면 추가
    if state.has("selected_context_files") and state.get("selected_context_files"):
        selected_files = state.get("selected_context_files")
        context_content = "\n\n추가 컨텍스트 정보:\n"

        for file_path in selected_files:
            # extras/ 이후의 경로만 표시
            display_name = file_path.replace("extras/", "", 1)
            content = read_file_content(file_path)
            context_content += f"\n--- {display_name} ---\n{content}\n"

        base_prompt += context_content

    return base_prompt


# 초기 스키마 정보 가져오기
db_schema = get_db_schema()
if not db_schema:
    db_schema = "데이터베이스 연결에 실패했거나 스키마 정보가 없습니다."

# 상태 초기화
initialize_state()

# 사이드바에 스키마 정보와 추가 컨텍스트 선택 UI 표시
with st.sidebar:

    # 모델 선택기 컴포넌트 사용
    model_selector(state)

    # DB 스키마 정보
    with st.expander("데이터베이스 스키마 정보"):
        if st.button("DB 스키마 다시 조회"):
            with st.spinner("데이터베이스 스키마 정보를 조회중입니다..."):
                refreshed_schema = get_db_schema()
                if refreshed_schema:
                    db_schema = refreshed_schema
                    if update_system_message():
                        st.success("스키마 정보가 업데이트되었습니다.")
                        st.rerun()  # 페이지 새로고침으로 업데이트된 스키마 표시
                else:
                    st.error("데이터베이스 스키마를 가져오는데 실패했습니다.")

        # 스키마 정보 표시
        st.code(db_schema)

    # 추가 컨텍스트 파일 선택 UI 사용
    if context_file_selector(state):
        # 시스템 프롬프트 업데이트 버튼
        if st.button("컨텍스트 적용"):
            if update_system_message():
                st.success("추가 컨텍스트가 적용되었습니다.")
                st.rerun()


# 대화 초기화 버튼
if st.button("대화 초기화"):
    # 시스템 메시지를 제외한 모든 메시지 삭제
    full_system_prompt = build_prompt_context()
    state.set("sql_messages", [{"role": "system", "content": full_system_prompt}])
    state.set("generated_sql", "")
    st.rerun()  # 페이지 새로고침

# 구분선
st.divider()

# 채팅 기록 표시 탭과 SQL 실행 결과 탭
tab1, tab2 = st.tabs(["대화 기록", "SQL 실행 결과"])

with tab1:
    # 기존 채팅 기록 표시 (시스템 메시지는 표시하지 않음)
    if state.has("sql_messages"):
        for message in state.get("sql_messages"):
            if message["role"] != "system":
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

# 사용자 입력
prompt = st.chat_input("SQL로 변환할 요청을 입력하세요 (예: 모든 데이터를 조회해줘)")
if prompt:
    # 사용자 메시지 추가
    messages = state.get("sql_messages")
    messages.append({"role": "user", "content": prompt})
    state.set("sql_messages", messages)

    # 화면에 즉시 표시
    with tab1:
        with st.chat_message("user"):
            st.markdown(prompt)

    # AI 응답 표시
    with tab1:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            # 추적 기능이 포함된 함수 호출
            stream = tracked_chat_completion(
                model=state.selected_model,
                messages=state.get("sql_messages"),
                stream=True,
            )

            # 응답 처리
            full_response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)

            # 응답 저장
            messages = state.get("sql_messages")
            messages.append({"role": "assistant", "content": full_response})
            state.set("sql_messages", messages)

            cleaned_response = clean_sql_response(full_response)

            # SQL이 생성되지 않은 경우 적절한 처리
            if not cleaned_response:
                message_placeholder.markdown(
                    "SQL 쿼리를 생성할 수 없습니다. 질문을 더 명확하게 해주세요."
                )

            state.set("generated_sql", cleaned_response)

# SQL 실행 섹션
with tab2:
    st.subheader("SQL 쿼리 실행")

    # 생성된 쿼리 표시 및 편집 가능하게
    generated_sql = state.get("generated_sql", "")

    if not generated_sql:
        st.warning(
            "실행할 SQL 쿼리가 없습니다. 질문을 다시 입력하거나 직접 SQL을 작성하세요."
        )

    sql_query = st.text_area("실행할 SQL 쿼리", value=generated_sql, height=150)

    # 쿼리 실행 버튼
    if st.button("쿼리 실행"):
        if not sql_query.strip():
            st.warning("실행할 SQL 쿼리를 입력하세요.")
        else:
            with st.spinner("쿼리 실행 중..."):
                results, error = execute_query(sql_query)

            # 컴포넌트를 사용하여 결과 표시
            display_query_results(results, error)
