from typing import Dict, List, Any, Optional, TypedDict
import streamlit as st


# 메시지 타입 정의
class Message(TypedDict):
    """대화 메시지 타입 정의"""

    role: str  # 'user' 또는 'assistant'
    content: str  # 메시지 내용


class AppState:
    """애플리케이션 상태를 관리하는 클래스

    Streamlit의 session_state를 래핑하여 타입 안정성과 속성 접근 편의성을 제공합니다.
    """

    def __init__(self, session_state) -> None:
        """
        session_state: List[Message] 타입의 메시지 목록
        """
        self._session_state = session_state # List[Message]

        # 기본값 초기화
        self._init_default_values()

    def _init_default_values(self) -> None:
        """기본 상태값 초기화"""
        if "messages" not in self._session_state:
            self._session_state.messages = []

        if "selected_model" not in self._session_state:
            self._session_state.selected_model = "gpt-4o"

        if "available_models" not in self._session_state:
            self._session_state.available_models = []

    # 메시지 관련 속성
    @property
    def messages(self) -> List[Message]:
        """대화 메시지 목록

        @property 데코레이터의 역할:
        - 메서드를 속성처럼 접근할 수 있게 해줍니다.
        - 외부에서는 state.messages와 같이 속성처럼 접근하지만, 내부적으로는 메서드가 호출됩니다.
        - getter 역할을 수행하여 session_state의 값을 반환합니다.
        - 이를 통해 데이터 접근에 대한 일관된 인터페이스를 제공합니다.
        """
        return self._session_state.messages

    @messages.setter
    def messages(self, value: List[Message]):
        """대화 메시지 목록 설정

        @messages.setter 데코레이터의 역할:
        - 속성에 값을 할당할 때 호출되는 메서드를 정의합니다. (state.messages = [...] 형태로 사용)
        - setter 역할을 수행하여 값을 검증하거나 변환한 후 session_state에 저장할 수 있습니다.
        - property와 함께 사용하여 완전한 getter/setter 패턴을 구현합니다.
        - 캡슐화를 제공하여 내부 구현 세부사항을 숨깁니다.
        """
        self._session_state.messages = value

    def add_message(self, role: str, content: str) -> None:
        """메시지 추가 헬퍼 메서드"""
        self._session_state.messages.append({"role": role, "content": content})

    def clear_messages(self) -> None:
        """모든 메시지 지우기"""
        self._session_state.messages = []

    # 모델 관련 속성
    @property
    def selected_model(self) -> str:
        """현재 선택된 모델"""
        return self._session_state.selected_model

    @selected_model.setter
    def selected_model(self, value: str):
        """선택된 모델 설정"""
        self._session_state.selected_model = value

    @property
    def available_models(self) -> List[str]:
        """사용 가능한 모델 목록"""
        return self._session_state.available_models

    @available_models.setter
    def available_models(self, value: List[str]):
        """사용 가능한 모델 목록 설정"""
        self._session_state.available_models = value

    # 일반적인 세션 상태 접근 메서드
    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """키에 해당하는 값 가져오기"""
        return self._session_state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """키에 값 설정하기"""
        self._session_state[key] = value

    def has(self, key: str) -> bool:
        """키가 존재하는지 확인"""
        return key in self._session_state
