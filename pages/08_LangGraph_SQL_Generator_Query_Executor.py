import streamlit as st
import pandas as pd
import os
import json
import glob
from typing import List, Dict, Any, Tuple

# 커스텀 모듈 임포트
from text2sql import AppState
from text2sql.openai_utils import initialize_models, refresh_models
from text2sql.db_utils import execute_query, get_all_tables, get_table_schema
from text2sql.graphs.sql_graph import sql_graph

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


# extras 디렉토리에서 파일 목록 가져오기
def get_extra_files():
    # 메인 extras 디렉토리의 파일 목록
    md_files = glob.glob("extras/*.md")
    md_files = [f for f in md_files if os.path.isfile(f)]

    # E_commerce 디렉토리의 파일 목록
    ecom_files = glob.glob("extras/E_commerce/*.*")
    ecom_files = [f for f in ecom_files if os.path.isfile(f)]

    # 모든 파일 목록 합치기
    all_files = md_files + ecom_files

    # 파일 경로에서 파일명만 추출
    file_options = {}
    for file_path in all_files:
        file_name = os.path.basename(file_path)
        file_options[file_name] = file_path

    return file_options


# 파일 내용 읽기 함수
def read_file_content(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            # 파일 확장자에 따라 처리 방식 분기
            if file_path.endswith(".json"):
                try:
                    return json.dumps(json.load(file), indent=2, ensure_ascii=False)
                except:
                    return file.read()  # JSON 파싱 실패시 원본 텍스트 반환
            else:
                # 기타 파일은 텍스트로 읽음
                return file.read()
    except Exception as e:
        return f"파일 읽기 오류: {str(e)}"


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
        file_name = os.path.basename(file_path)
        content = read_file_content(file_path)
        context += f"\n--- {file_name} ---\n{content}\n"

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

    # 추가 컨텍스트 파일 선택 UI
    with st.expander("추가 컨텍스트 파일 선택"):
        # 파일 목록 가져오기
        file_options = get_extra_files()

        # 파일 선택 UI
        selected_files = st.multiselect(
            "컨텍스트로 사용할 파일을 선택하세요",
            options=list(file_options.keys()),
            default=[
                os.path.basename(f)
                for f in state.get("selected_context_files")
                if os.path.basename(f) in file_options.keys()
            ],
            help="선택한 파일의 내용이 SQL 생성 프롬프트에 추가됩니다.",
        )

        # 선택된 파일 경로 저장
        selected_file_paths = [
            file_options[file_name]
            for file_name in selected_files
            if file_name in file_options
        ]
        state.set("selected_context_files", selected_file_paths)

        # 선택된 파일 미리보기
        if selected_file_paths:
            st.write("선택된 파일 미리보기:")
            for file_path in selected_file_paths:
                file_name = os.path.basename(file_path)
                st.markdown(f"**{file_name}**")
                container = st.container()
                with container:
                    content = read_file_content(file_path)
                    # JSON이나 긴 텍스트는 코드 블록으로 표시
                    if file_path.endswith(".json") or len(content) > 500:
                        st.code(content[:2000] + ("..." if len(content) > 2000 else ""))
                    else:
                        st.write(
                            content[:2000] + ("..." if len(content) > 2000 else "")
                        )
                st.markdown("---")  # 구분선 추가

            # 컨텍스트 적용 버튼
            if st.button("컨텍스트 적용"):
                update_context()
                st.success("추가 컨텍스트가 적용되었습니다.")
                st.rerun()

# 모델 정보 표시 (접기 가능)
with st.expander("현재 설정"):
    st.write(f"현재 모델: {state.selected_model}")

    # 모델 목록 새로고침 버튼
    if st.button("모델 목록 새로고침"):
        if refresh_models(state):
            st.success("모델 목록이 업데이트되었습니다!")
            st.rerun()  # 페이지 새로고침
        else:
            st.error("모델 목록 업데이트에 실패했습니다.")

# 모델 선택 UI
if state.available_models:
    selected_index = (
        state.available_models.index(state.selected_model)
        if state.selected_model in state.available_models
        else 0
    )
    selected_model = st.selectbox("모델:", state.available_models, index=selected_index)
    state.selected_model = selected_model  # 선택한 모델 저장

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

        # 결과 처리
        if (
            not query_result or len(query_result) <= 1
        ):  # 컬럼명만 있고 데이터가 없는 경우
            st.info("쿼리가 성공적으로 실행되었지만 반환된 결과가 없습니다.")
        else:
            # 성공적인 결과 표시
            row_count = len(query_result) - 1  # 첫 번째 행은 컬럼명
            st.success(f"쿼리 실행 성공: {row_count}개의 결과를 찾았습니다.")

            # 첫 번째 행을 컬럼명으로 사용
            columns = query_result[0]
            # 첫 번째 행을 제외한 나머지를 데이터로 사용
            data = query_result[1:]

            # 결과를 데이터프레임으로 변환하여 표시
            df = pd.DataFrame(data, columns=columns)
            st.dataframe(df)

            # CSV 다운로드 버튼 추가
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "CSV로 다운로드",
                csv,
                "query_results.csv",
                "text/csv",
                key="download-csv",
            )

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

                # 새로운 응답이 있다면 history 업데이트
                if error_result.get("llm_response"):
                    state.set("sql_history", error_result["history"])
                    # 세션 상태에도 직접 저장
                    st.session_state.sql_history = error_result["history"]

            # 결과 처리
            elif not results or len(results) <= 1:  # 컬럼명만 있고 데이터가 없는 경우
                st.info("쿼리가 성공적으로 실행되었지만 반환된 결과가 없습니다.")
            else:
                # 성공적인 결과 표시
                row_count = len(results) - 1  # 첫 번째 행은 컬럼명
                st.success(f"쿼리 실행 성공: {row_count}개의 결과를 찾았습니다.")

                # 첫 번째 행을 컬럼명으로 사용
                columns = results[0]
                # 첫 번째 행을 제외한 나머지를 데이터로 사용
                data = results[1:]

                # 결과를 데이터프레임으로 변환하여 표시
                df = pd.DataFrame(data, columns=columns)
                st.dataframe(df)

                # CSV 다운로드 버튼 추가
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "CSV로 다운로드",
                    csv,
                    "query_results.csv",
                    "text/csv",
                    key="download-csv-manual",
                )

                # LangGraph 상태에 성공 정보 추가
                langgraph_state = {
                    "question": "쿼리 실행 결과",
                    "schema": state.get("db_schema", ""),
                    "context": state.get("context", ""),
                    "sql": sql_query,
                    "error": None,
                    "result": results,  # 실제 결과 전달
                    "history": state.get("sql_history", []),
                    "model": state.selected_model,
                    "llm_response": None,  # LLM 응답 초기화
                }

                # 성공 정보를 LangGraph로 전달
                success_result = sql_graph.invoke(langgraph_state)

                # 새로운 응답이 있다면 history 업데이트
                if success_result.get("llm_response"):
                    state.set("sql_history", success_result["history"])
                    # 세션 상태에도 직접 저장
                    st.session_state.sql_history = success_result["history"]

                state.set("query_result", results)
                state.set("query_error", None)
