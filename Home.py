import streamlit as st

st.set_page_config(
    page_title="텍스트 to SQL 앱",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("텍스트 to SQL 앱")
st.markdown("---")

st.header("텍스트 to SQL 앱 소개")

# 메인 페이지 설명 추가
st.write(
    """
이 애플리케이션은 텍스트를 SQL로 변환하는 다양한 기능 제공.
각 페이지 설명:
"""
)

# 예제 페이지 설명 (01~06)
st.subheader("예제 페이지 (01~06)")
example_pages = {
    "01_Simple_Chat": "간단한 챗봇 - OpenAI API 사용 기본 챗 인터페이스.",
    "02_Simple_Query_Generator": "간단한 SQL 생성기 - 한국어 질문을 SQL 쿼리로 변환.",
    "03_Simple_LangGraph": "간단한 LangGraph - LangGraph 활용 기본 예제.",
    "04_Simple_LangGraph_With_LLM": "LLM이 포함된 LangGraph - LangGraph에 LLM 통합 예제.",
    "05_Simple_LangGraph_With_History": "대화 기록이 포함된 LangGraph - 대화 기록 유지하는 LangGraph 예제.",
    "06_Simple_Database": "간단한 데이터베이스 - SQL 쿼리 실행 및 결과 확인 가능.",
}

for page, description in example_pages.items():
    st.markdown(f"**{page}**: {description}")

# 실전 애플리케이션 설명 (07~08)
st.subheader("애플리케이션 (07~08)")

st.markdown(
    """
**07_SQL_Generator_Query_Executor**: 스트리밍 기반 SQL 생성 및 실행기
- 자연어를 SQL로 변환 및 실행하는 애플리케이션.
- 데이터베이스 스키마 정보 활용.
- 추가 컨텍스트 파일 선택하여 SQL 생성에 활용 가능.
- 생성된 SQL 실행 및 결과 확인.
"""
)

st.markdown(
    """
**08_LangGraph_SQL_Generator_Query_Executor**: LangGraph 에이전트를 활용한 SQL 생성 및 실행기
- LangGraph 활용.
- 단계별 처리(질문 이해, SQL 생성, SQL 검증, 쿼리 실행, 결과 해석).
- 복잡한 대화 흐름과 오류 처리 지원.
- 프롬프트 엔지니어링과 대화형 SQL 생성 가능.
"""
)

# 시작하기 안내
st.markdown("---")
st.subheader("시작하기")


# 세션 상태 초기화
