"""
OpenAI API 사용을 위한 유틸리티 모듈
"""

import os
import streamlit as st
from openai import OpenAI
from langsmith import traceable
from text2sql.state import AppState


def initialize_env_vars():
    """LangSmith 환경 변수를 설정합니다."""
    os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
    os.environ["LANGCHAIN_TRACING_V2"] = st.secrets["LANGCHAIN_TRACING_V2"]
    os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_PROJECT"]


# LangSmith 환경 변수 설정
initialize_env_vars()

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


@traceable(name="openai_chat_completion", run_type="llm")
def tracked_chat_completion(model, messages, stream=True):
    """
    OpenAI 채팅 완료 API를 호출하고 LangSmith로 추적합니다.
    LLM 유형으로 추적하며 토큰 수를 기록합니다.

    Args:
        model (str): 사용할 OpenAI 모델 ID
        messages (list): 대화 메시지 목록
        temperature (float): 생성 온도 (0~1)
        stream (bool): 스트리밍 모드 사용 여부

    Returns:
        OpenAI 응답 객체 또는 스트림
    """
    return client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        seed=0,
        top_p=1,
        stream=stream,
    )


def get_available_models():
    """
    OpenAI API에서 사용 가능한 GPT 모델 목록을 가져옵니다.

    Returns:
        list: 모델 ID 목록
    """
    try:
        models = client.models.list()
        # chat completion 모델만 필터링
        chat_models = [model.id for model in models if "gpt" in model.id]
        chat_models.sort()
        return chat_models
    except Exception as e:
        st.error(f"모델 목록을 가져오는데 실패했습니다: {str(e)}")
        return ["gpt-4o"]  # 기본값


def initialize_models(state: AppState):
    """
    애플리케이션 상태에서 사용 가능한 모델 목록을 초기화하고 현재 선택된 모델이 유효한지 확인합니다.

    Args:
        state: AppState 인스턴스

    Returns:
        bool: 모델 목록이 업데이트되었으면 True, 아니면 False
    """
    updated = False

    # 모델 목록이 비어있는 경우에만 가져오기
    if not state.available_models:
        state.available_models = get_available_models()

    # 모델 목록이 여전히 비어있으면 기본값 설정
    if not state.available_models:
        state.available_models = ["gpt-4o"]

    # 선택된 모델이 목록에 없으면 첫 번째 모델로 설정
    if state.selected_model not in state.available_models:
        state.selected_model = state.available_models[0]

    return updated


def refresh_models(state):
    """
    모델 목록을 새로고침하고 애플리케이션 상태를 업데이트합니다.

    Args:
        state: AppState 인스턴스

    Returns:
        bool: 성공적으로 새로고침되었으면 True, 아니면 False
    """
    try:
        state.available_models = get_available_models()

        # 선택된 모델이 유효한지 확인
        if (
            state.selected_model not in state.available_models
            and state.available_models
        ):
            state.selected_model = state.available_models[0]

        return True
    except Exception as e:
        st.error(f"모델 목록 새로고침 중 오류 발생: {str(e)}")
        return False
