"""
OpenAI API 사용을 위한 유틸리티 모듈
"""

import os
import streamlit as st
from openai import OpenAI
from langsmith import traceable


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
