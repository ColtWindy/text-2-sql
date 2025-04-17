import streamlit as st
from typing import List, Any, Tuple

# 커스텀 모듈 임포트
from text2sql import AppState
from text2sql.openai_utils import initialize_models, refresh_models
from text2sql.db_utils import execute_query, get_all_tables, get_table_schema
from text2sql.graphs.sql_graph import sql_graph
from text2sql.components import (
    model_selector,
    read_file_content,
    context_file_selector,
    display_query_results,
)

# 페이지 설정
st.title("LangGraph SQL 생성 및 실행기")
st.write("데이터 분석가와 오퍼레이터를 위한 자연어 기반 데이터베이스 조회 도구입니다.")

# 애플리케이션 상태 초기화
state = AppState(st.session_state)

# 모델 초기화 실행
initialize_models(state)


# LangGraph SQL 그래프용 상태 초기화 함수
def initialize_langgraph_state():
    """LangGraph 상태 변수를 초기화하는 함수"""
    if not state.has("sql_history"):
        state.set("sql_history", [])
    if not state.has("generated_sql"):
        state.set("generated_sql", "")
    if not state.has("db_schema"):
        state.set("db_schema", get_db_schema())
    if not state.has("context"):
        state.set("context", "")
    if not state.has("selected_context_files"):
        state.set("selected_context_files", [])
    if not state.has("llm_response"):
        state.set("llm_response", None)


# 데이터베이스 스키마 정보를 가져오는 함수
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

    return schema_text


# 컨텍스트 정보 업데이트 함수
def update_context():
    context = ""
    selected_files = state.get("selected_context_files")

    for file_path in selected_files:
        # extras/ 이후의 경로만 표시
        display_name = file_path.replace("extras/", "", 1)
        content = read_file_content(file_path)
        context += f"\n--- {display_name} ---\n{content}\n"

    state.set("context", context)
    return context


# SQL 실행 함수 (LangGraph의 EXECUTE_SQL 노드에서 사용)
def execute_sql_query(
    sql_query: str,
) -> Tuple[List[Tuple[Any, ...]] | None, str | None]:
    """SQL 쿼리를 실행하고 결과 및 오류를 반환"""
    results, error = execute_query(sql_query)
    return results, error


# 상태 초기화
initialize_langgraph_state()

# 사이드바에 스키마 정보와 추가 컨텍스트 선택 UI 표시
with st.sidebar:
    # 모델 선택기 컴포넌트 사용
    model_selector(state)

    # DB 스키마 정보
    with st.expander("데이터베이스 스키마 정보"):
        # 데이터베이스 테이블 정보 동적으로 가져오기
        if st.button("DB 스키마 다시 조회"):
            with st.spinner("데이터베이스 스키마 정보를 조회중입니다..."):
                refreshed_schema = get_db_schema()
                if refreshed_schema:
                    state.set("db_schema", refreshed_schema)
                    st.success("스키마 정보가 업데이트되었습니다.")
                    st.rerun()
                else:
                    st.error("데이터베이스 스키마를 가져오는데 실패했습니다.")

        # 스키마 정보 표시
        st.code(state.get("db_schema", ""))

    # 추가 컨텍스트 파일 선택 UI 사용
    if context_file_selector(state):
        # 컨텍스트 적용 버튼
        if st.button("컨텍스트 적용"):
            update_context()
            st.success("추가 컨텍스트가 적용되었습니다.")
            st.rerun()


# 대화 초기화 버튼
if st.button("대화 초기화"):
    state.set("sql_history", [])
    state.set("generated_sql", "")
    st.session_state.sql_history = []  # 세션 상태에도 직접 추가
    st.rerun()  # 페이지 새로고침

# 구분선
st.divider()

# 채팅 기록 표시 탭과 SQL 실행 결과 탭
tab1, tab2 = st.tabs(["대화 기록", "SQL 실행 결과"])

with tab1:
    # 기존 채팅 기록 표시
    for message in state.get("sql_history", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# 사용자 입력
prompt = st.chat_input("SQL로 변환할 요청을 입력하세요 (예: 모든 데이터를 조회해줘)")
if prompt:
    # 화면에 즉시 표시
    with tab1:
        with st.chat_message("user"):
            st.markdown(prompt)

    # 사용자 메시지 직접 추가하지 않고 LangGraph가 처리하도록 함
    history = state.get("sql_history", [])

    # LangGraph 그래프 호출을 위한 초기 상태 설정
    initial_state = {
        "question": prompt,
        "schema": state.get("db_schema", ""),
        "context": state.get("context", ""),
        "sql": "",
        "error": None,
        "result": None,
        "history": history,  # 기존 히스토리 그대로 전달
        "model": state.selected_model,
        "llm_response": None,  # LLM 응답 초기화
    }

    # AI 응답 표시
    with tab1:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            # 스피너와 함께 진행 상태 표시
            with st.spinner("LangGraph 처리 중..."):
                # SQL 생성 그래프 호출
                result = sql_graph.invoke(initial_state)

            # 결과에서 SQL, LLM 응답 및 실행 결과 추출
            generated_sql = result["sql"]
            llm_response = result.get("llm_response")
            query_result = result.get("result")
            query_error = result.get("error")

            # LLM 전체 응답 표시 (있는 경우)
            if llm_response:
                message_placeholder.markdown(llm_response)
            else:
                # LLM 응답이 없으면 생성된 SQL만 표시
                message_placeholder.markdown(f"```sql\n{generated_sql}\n```")

                # 결과가 있으면 함께 표시
                if query_result and len(query_result) > 1:
                    row_count = len(query_result) - 1
                    message_placeholder.markdown(
                        f"쿼리 실행 결과: {row_count}개의 행이 조회되었습니다."
                    )

            # 생성된 SQL 저장
            state.set("generated_sql", generated_sql)

            # 실행 결과 저장
            state.set("query_result", query_result)
            state.set("query_error", query_error)

            # 히스토리 업데이트 - LangGraph에서 반환된 히스토리 사용
            state.set("sql_history", result["history"])
            # 세션 상태에도 직접 설정하여 페이지 이동 후에도 유지되도록 함
            st.session_state.sql_history = result["history"]

            # 실행 결과가 있으면 결과 탭으로 전환
            if query_result and not query_error:
                st.rerun()  # 페이지 새로고침

# SQL 실행 섹션
with tab2:
    st.subheader("SQL 쿼리 실행")

    # 자동으로 생성된 SQL 쿼리와 실행 결과 가져오기
    generated_sql = state.get("generated_sql", "")
    query_result = state.get("query_result")
    query_error = state.get("query_error")

    if not generated_sql:
        st.warning(
            "실행할 SQL 쿼리가 없습니다. 질문을 다시 입력하거나 직접 SQL을 작성하세요."
        )

    # SQL 쿼리 표시 및 편집 가능하게
    sql_query = st.text_area("실행할 SQL 쿼리", value=generated_sql, height=150)

    # 자동 실행 결과 표시 (있는 경우)
    if query_result and not query_error:
        st.success("자동 SQL 실행 결과:")
        # 컴포넌트를 사용하여 결과 표시
        display_query_results(query_result, None)

    # 자동 실행 중 오류가 발생한 경우
    elif query_error:
        st.error(f"자동 SQL 실행 실패: {query_error}")
        st.warning("SQL 쿼리를 수정하고 다시 실행해보세요.")

    # 쿼리 수동 실행 버튼
    if st.button("쿼리 수동 실행"):
        if not sql_query.strip():
            st.warning("실행할 SQL 쿼리를 입력하세요.")
        else:
            # 쿼리 실행
            with st.spinner("쿼리 실행 중..."):
                results, error = execute_query(sql_query)

            # 오류 처리
            if error is not None:
                st.error(f"쿼리 실행 실패: {error}")
                st.warning("SQL 쿼리를 수정하고 다시 시도해보세요.")

                # LangGraph 상태에 오류 정보 추가 (다음 대화에 활용)
                langgraph_state = {
                    "question": "이전 쿼리 오류 수정",
                    "schema": state.get("db_schema", ""),
                    "context": state.get("context", ""),
                    "sql": sql_query,
                    "error": str(error),
                    "result": None,
                    "history": state.get("sql_history", []),
                    "model": state.selected_model,
                    "llm_response": None,  # LLM 응답 초기화
                }

                # 오류 정보를 LangGraph로 전달하고 처리 결과 가져오기
                error_result = sql_graph.invoke(langgraph_state)

                if error_result.get("sql") and error_result["sql"] != sql_query:
                    st.success("수정된 SQL 쿼리를 생성했습니다:")
                    st.code(error_result["sql"], language="sql")
                    st.button(
                        "수정된 쿼리 적용",
                        on_click=lambda: st.session_state.update(
                            {"fixed_sql": error_result["sql"]}
                        ),
                    )

            else:
                # 컴포넌트를 사용하여 결과 표시 (수동 쿼리의 경우 접미사 추가)
                display_query_results(results, None, "manual")
