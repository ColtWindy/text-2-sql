"""
재사용 가능한 Streamlit 컴포넌트 모듈
"""

import streamlit as st
import glob
import os
import json
from text2sql.openai_utils import refresh_models
from text2sql.state import AppState


def model_selector(state):
    """
    사이드바에 모델 선택기 컴포넌트를 표시합니다.

    Args:
        state: AppState 인스턴스

    Returns:
        str: 선택된 모델 이름
    """
    with st.sidebar:
        with st.expander("🤖 모델 설정", expanded=False):
            st.write(f"현재 모델: **{state.selected_model}**")

            # 모델 목록 새로고침 버튼
            if st.button("모델 목록 새로고침"):
                if refresh_models(state):
                    st.success("모델 목록이 업데이트되었습니다!")
                    st.rerun()  # 페이지 새로고침
                else:
                    st.error("모델 목록 업데이트에 실패했습니다.")

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

    return state.selected_model


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
