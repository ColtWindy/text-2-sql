"""
재사용 가능한 Streamlit 컴포넌트 모듈
"""

import streamlit as st
import glob
import os
import json
import pandas as pd
from typing import List, Optional, Tuple
from text2sql.state import DB_CONFIGS, AppState
from text2sql.db_utils import get_all_tables, get_table_schema


def model_selector(state: AppState):
    """
    사이드바에 모델 선택기와 DB 설정 컴포넌트를 표시합니다.

    Args:
        state: AppState 인스턴스

    Returns:
        str: 선택된 모델 이름
    """
    with st.sidebar:
        with st.expander("⚙️ 환경 설정", expanded=False):
            # 모델 설정
            st.write("#### 모델 설정")

            # 모델 선택 드롭다운
            if state.available_models:
                selected_index = (
                    state.available_models.index(state.selected_model)
                    if state.selected_model in state.available_models
                    else 0
                )
                selected_model = st.selectbox(
                    "사용할 모델:", state.available_models, index=selected_index
                )

                # 모델이 변경되었으면 저장
                if selected_model != state.selected_model:
                    state.selected_model = selected_model
                    st.success(f"모델이 {selected_model}로 변경되었습니다.")
            else:
                st.warning("사용 가능한 모델이 없습니다.")

            # DB 설정 부분을 별도 함수 호출로 분리
            db_selector(state)

    return state.selected_model


def db_selector(state: AppState):
    """
    데이터베이스 선택 컴포넌트를 표시합니다.

    Args:
        state: AppState 인스턴스

    Returns:
        int: 현재 선택된 DB 인덱스
    """
    # DB 설정
    st.write("#### 데이터베이스 설정")

    # 현재 DB 설정 정보
    current_db_config = state.current_db_config
    current_db_name = current_db_config.name

    st.write(f"현재 DB: **{current_db_name}**")

    # DB 목록에서 이름만 추출
    db_names = [db.name for db in DB_CONFIGS]

    # DB 선택 드롭다운
    selected_db_name = st.selectbox(
        "사용할 데이터베이스:", options=db_names, index=state.selected_db_index
    )

    # 선택된 이름에 해당하는 인덱스 찾기
    selected_index = next(
        (i for i, name in enumerate(db_names) if name == selected_db_name),
        0,  # 기본값은 첫 번째 DB
    )

    # DB가 변경되었으면 저장
    if selected_index != state.selected_db_index:
        state.selected_db_index = selected_index
        st.success(f"데이터베이스가 {selected_db_name}로 변경되었습니다.")
        st.info("DB 스키마를 다시 조회해야 합니다.")

    return selected_index


def get_extra_files():
    """
    extras 디렉토리에서 모든 파일 목록을 가져옵니다.

    Returns:
        dict: {표시이름: 파일경로} 형태의 딕셔너리
    """
    # glob 패턴을 사용하여 extras 디렉토리의 모든 파일 검색
    all_files = glob.glob("extras/**/*.*", recursive=True)
    all_files = [f for f in all_files if os.path.isfile(f)]

    # 파일 옵션 딕셔너리 생성
    file_options = {}
    for file_path in all_files:
        # extras/ 이후의 경로만 표시
        display_name = file_path.replace("extras/", "", 1)
        file_options[display_name] = file_path

    return file_options


def read_file_content(file_path):
    """
    파일 내용을 읽어 반환합니다.

    Args:
        file_path: 읽을 파일 경로

    Returns:
        str: 파일 내용
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
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


def context_file_selector(state: AppState):
    """
    추가 컨텍스트 파일 선택 UI를 표시합니다.

    Args:
        state: AppState 인스턴스

    Returns:
        str: 선택된 파일들의 컨텍스트 내용
    """
    with st.expander("추가 컨텍스트 파일 선택"):
        # 파일 목록 가져오기
        file_options = get_extra_files()

        # 파일 선택 UI
        selected_files = st.multiselect(
            "컨텍스트로 사용할 파일을 선택하세요",
            options=list(file_options.keys()),
            default=[
                f.replace("extras/", "", 1)
                for f in state.get("selected_context_files", [])
                if f.replace("extras/", "", 1) in file_options.keys()
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
                # extras/ 이후의 경로만 표시
                display_name = file_path.replace("extras/", "", 1)
                st.markdown(f"**{display_name}**")
                container = st.container()
                with container:
                    content = read_file_content(file_path)
                    # 모든 파일을 코드 블록으로 표시
                    st.code(content[:2000] + ("..." if len(content) > 2000 else ""))
                st.markdown("---")  # 구분선 추가

            return True

        return False


def display_query_results(
    results: Optional[List[Tuple]], error: Optional[str], key_suffix: str = ""
):
    """
    SQL 쿼리 실행 결과를 화면에 표시하고 CSV 다운로드 버튼을 제공합니다.

    Args:
        results: SQL 쿼리 실행 결과 (첫 번째 행은 컬럼명)
        error: 에러 메시지 (에러가 없으면 None)
        key_suffix: 다운로드 버튼 키의 접미사 (여러 버튼 구분용)

    Returns:
        bool: 쿼리 실행 성공 여부
    """
    if error is not None:
        st.error(f"쿼리 실행 실패: {error}")
        st.warning("SQL 쿼리를 수정하고 다시 시도해보세요.")
        return False

    elif not results or len(results) <= 1:  # 컬럼명만 있고 데이터가 없는 경우
        st.info("쿼리가 성공적으로 실행되었지만 반환된 결과가 없습니다.")
        return True

    else:
        # 성공적인 결과 표시
        row_count = len(results) - 1  # 첫 번째 행은 컬럼명
        st.success(f"쿼리 실행 성공: {row_count}개의 결과를 찾았습니다.")

        # 결과를 데이터프레임으로 변환하여 표시
        columns = results[0]
        data = results[1:]
        df = pd.DataFrame(data, columns=columns)
        st.dataframe(df)

        # CSV 다운로드 버튼 추가
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "CSV로 다운로드",
            csv,
            "query_results.csv",
            "text/csv",
            key=f"download-csv{'-' + key_suffix if key_suffix else ''}",
        )

        return True


def get_db_schema(state: AppState) -> str:
    """
    데이터베이스 스키마 정보를 가져옵니다.

    Args:
        state: AppState 인스턴스

    Returns:
        str: 스키마 정보 텍스트
    """
    schema_text = ""
    tables = get_all_tables(state)
    if tables:
        for table in tables:
            schema_text += f"\n테이블: {table}\n"

            columns = get_table_schema(table, state)
            if columns:
                for col in columns:
                    col_name = col[0]
                    data_type = col[1]
                    nullable = "NULL 허용" if col[2] == "YES" else "NOT NULL"
                    schema_text += f"- {col_name} ({data_type}, {nullable})\n"

    return schema_text


def load_db_schema(state: AppState, refresh: bool = False) -> str:
    """
    데이터베이스 스키마를 로드하거나 새로고침합니다.

    Args:
        state: AppState 인스턴스
        refresh: 강제로 새로고침할지 여부

    Returns:
        str: 현재 스키마 정보
    """
    # 기존 스키마가 없거나 새로고침 요청이 있으면 새로 가져옴
    if refresh or not state.has("db_schema") or not state.get("db_schema"):
        schema = get_db_schema(state)
        if not schema:
            schema = "데이터베이스 연결에 실패했거나 스키마 정보가 없습니다."
        state.set("db_schema", schema)
        return schema

    # 이미 있는 스키마 반환
    return state.get("db_schema")


def update_context_from_files(state: AppState) -> str:
    """
    선택된 컨텍스트 파일에서 컨텍스트 정보를 업데이트합니다.

    Args:
        state: AppState 인스턴스

    Returns:
        str: 업데이트된 컨텍스트 정보
    """
    context = ""
    selected_files = state.get("selected_context_files", [])

    for file_path in selected_files:
        # extras/ 이후의 경로만 표시
        display_name = file_path.replace("extras/", "", 1)
        content = read_file_content(file_path)
        context += f"\n--- {display_name} ---\n{content}\n"

    state.set("context", context)
    return context


def db_schema_expander(state: AppState) -> bool:
    """
    데이터베이스 스키마 정보 확장 패널을 표시합니다.

    Args:
        state: AppState 인스턴스

    Returns:
        bool: 스키마가 새로고침되었는지 여부
    """
    with st.expander("데이터베이스 스키마 정보"):
        # 현재 DB 스키마 로드
        current_schema = load_db_schema(state)

        # 데이터베이스 테이블 정보 동적으로 가져오기
        refreshed = False
        if st.button("DB 스키마 다시 조회"):
            with st.spinner("데이터베이스 스키마 정보를 조회중입니다..."):
                refreshed_schema = load_db_schema(state, refresh=True)
                if refreshed_schema:
                    st.success("스키마 정보가 업데이트되었습니다.")
                    refreshed = True
                else:
                    st.error("데이터베이스 스키마를 가져오는데 실패했습니다.")

        # 스키마 정보 표시
        st.code(current_schema)

        return refreshed
