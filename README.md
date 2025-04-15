## 프로젝트 설명

- LLM 스터디 TEXT2SQL 발표를 위한 저장소입니다
- [TEXT2SQL](TEXT2SQL.md) 참고해 주세요

## 설정

LangGraph Studio는 옵션입니다. [Cli 설치](https://langchain-ai.github.io/langgraph/cloud/reference/cli/)가 필요합니다.

1. Streamlit: `.streamlit/secrets.toml` 파일에 다음을 작성합니다.

```toml
OPENAI_API_KEY = "..."
# LangSmith 설정
LANGCHAIN_API_KEY = "..."
LANGCHAIN_TRACING_V2 = "true"
LANGCHAIN_PROJECT = "text2sql"
```

2. LangGraph Studio: `.env` 에, `OPENAI_API_KEY` 설정

## 준비

[DB초기화 파일](https://drive.google.com/file/d/18jGW55_oilSF_rCQvtLftCSpPVSy6K8W/view?usp=sharing)를 다운받아, `database/` 폹더에 넣어주세요.

## 설치

- 의존성: `poetry install`
- 가상환경 진입: `poetry shell`
- docker(db): `docker compose up -d`

## 실행

- 앱 실행: `streamlit run Home.py`
- Langㅎraph Studio 실행(옵션): `poetry run langgraph dev`

## DB 정보

`docker-compose.yml` 파일 참고

## 소스 설명

- `extras/`: LLM 포함할 컨텍스트 정보
- `extras/E_commerce`: 데이터베이스 정보 및 예제 데이터
- `pages/`: 소스 코드
- `text2sql/graphs/`: 그래프 정의
- `langgraph.json`: LangGraph Studio 설정 파일
