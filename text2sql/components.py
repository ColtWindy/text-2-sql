"""
재사용 가능한 Streamlit 컴포넌트 모듈
"""

import streamlit as st
from text2sql.openai_utils import refresh_models


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
