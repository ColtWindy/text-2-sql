from typing import Dict, List, Any, Optional, TypedDict
from dataclasses import dataclass
from text2sql.openai_utils import get_available_models


# 메시지 타입 정의
class Message(TypedDict):
    """대화 메시지 타입 정의"""

    role: str  # 'user' 또는 'assistant'
    content: str  # 메시지 내용


# 데이터베이스 설정 타입 정의
@dataclass
class DBConfig:
    """데이터베이스 설정 타입 정의"""

    name: str = ""
    host: str = ""
    port: int = 0
    dbname: str = ""
    user: str = ""
    password: str = ""

    # 기본 데이터베이스 설정 목록 - 상수로 정의


DB_CONFIGS: List[DBConfig] = [
    DBConfig(
        name="E-Commerce",
        host="localhost",
        port=55433,
        dbname="postgres",
        user="postgres",
        password="postgres",
    ),
    DBConfig(
        name="E-Commerce (Nonintuitive)",
        host="localhost",
        port=55434,
        dbname="postgres",
        user="postgres",
        password="postgres",
    ),
    DBConfig(
        name="Text2SQL",
        host="localhost",
        port=55432,
        dbname="postgres",
        user="postgres",
        password="postgres",
    ),
]


class AppState:
    """애플리케이션 상태를 관리하는 클래스

    Streamlit의 session_state를 래핑하여 타입 안정성과 속성 접근 편의성을 제공합니다.
    """

    def __init__(self, session_state):
        """
        session_state: List[Message] 타입의 메시지 목록
        """
        self._session_state = session_state  # List[Message]

        # 기본값 초기화
        self._init_default_values()

    def _init_default_values(self):
        """기본 상태값 초기화"""
        if "messages" not in self._session_state:
            self._session_state.messages = []

        if "selected_model" not in self._session_state:
            self._session_state.selected_model = "gpt-4o"

        if "available_models" not in self._session_state:
            self._session_state.available_models = []

        # 현재 선택된 데이터베이스 인덱스 초기화 (기본값: 첫 번째 DB)
        if "selected_db_index" not in self._session_state:
            self._session_state.selected_db_index = 0

        # 모델과 DB 리소스 초기화
        self.initialize_resources()

    # 메시지 관련 속성
    @property
    def messages(self) -> List[Message]:
        """대화 메시지 목록"""
        return self._session_state.messages

    @messages.setter
    def messages(self, value: List[Message]):
        """대화 메시지 목록 설정"""
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

    # 데이터베이스 관련 속성 및 메서드
    @property
    def selected_db_index(self) -> int:
        """현재 선택된 데이터베이스 인덱스"""
        return self._session_state.selected_db_index

    @selected_db_index.setter
    def selected_db_index(self, value: int):
        """선택된 데이터베이스 인덱스 설정"""
        self._session_state.selected_db_index = value

    @property
    def current_db_config(self) -> DBConfig:
        """현재 선택된 데이터베이스 설정"""
        index = self.selected_db_index
        if 0 <= index < len(DB_CONFIGS):
            return DB_CONFIGS[index]
        # 기본값 반환 (첫 번째 DB)
        return DB_CONFIGS[0]

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

    def initialize_resources(self) -> bool:
        """
        애플리케이션의 리소스(모델, DB 설정)를 초기화합니다.

        Returns:
            bool: 리소스가 업데이트되었으면 True, 아니면 False
        """
        updated = False

        # 1. 모델 초기화
        # 모델 목록이 비어있는 경우에만 가져오기
        if not self.available_models:
            self.available_models = get_available_models()
            updated = True

        # 모델 목록이 여전히 비어있으면 기본값 설정
        if not self.available_models:
            self.available_models = ["gpt-4o"]
            updated = True

        # 선택된 모델이 목록에 없으면 첫 번째 모델로 설정
        if self.selected_model not in self.available_models:
            self.selected_model = self.available_models[0]
            updated = True

        # 2. DB 설정 초기화
        # 선택된 DB 인덱스가 유효한 범위인지 확인
        if self.selected_db_index < 0 or self.selected_db_index >= len(DB_CONFIGS):
            self.selected_db_index = 0
            updated = True

        # DB 스키마 정보 초기화 (없는 경우)
        if not self.has("db_schema") or not self.get("db_schema"):
            self.set("db_schema", "")
            updated = True

        # 연결 테스트 상태 초기화
        if not self.has("db_connection_status"):
            self.set("db_connection_status", None)
            updated = True

        return updated
