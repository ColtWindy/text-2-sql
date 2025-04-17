from typing import TypedDict, List, Any, Optional, Union
from dataclasses import dataclass, asdict, field
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage, BaseMessage
import re

from text2sql.db_utils import execute_query
from text2sql.state import DBConfig


# SQLState: 그래프의 상태를 정의하는 타입
@dataclass
class SQLState:
    """LangGraph 에이전트의 상태를 정의하는 데이터 클래스"""

    db: Optional[DBConfig] = None  # DB 설정
    question: str = ""  # 사용자 질문
    schema: str = ""  # DB 스키마 정보
    context: str = ""  # 추가 컨텍스트 정보
    sql: str = ""  # 생성된 SQL 쿼리
    error: Optional[str] = None  # SQL 실행 에러
    result: Any = None  # SQL 실행 결과
    # 파이썬에서, List[dict] = [] 와 같은 형식으로 초기화 할 때는 factory 키워드를 사용함
    history: List[dict] = field(default_factory=list)  # 대화 히스토리
    model: str = "gpt-4o"  # 사용할 모델명
    llm_response: Optional[str] = None  # LLM 응답
    retry_count: int = 0  # 재시도 횟수
    original_question: str = ""  # 원본 질문
    refine_count: int = 0  # 결과 평가 횟수
    summary: Optional[str] = None  # 결과 요약
    result_meta: Optional[str] = None  # 결과 메타 정보

    def __post_init__(self):
        # 초기화
        pass

    def to_dict(self) -> dict:
        """dataclass를 딕셔너리로 변환"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SQLState":
        """딕셔너리에서 dataclass 인스턴스 생성"""
        return cls(**data)


# 노드 이름 상수
class Node:
    GENERATE_SQL = "sql_generator"  # SQL 생성 노드
    EXECUTE_SQL = "sql_executor"  # SQL 실행 노드
    UPDATE_HISTORY = "history_updater"  # 히스토리 업데이트 노드
    HANDLE_ERROR = "error_handler"  # 에러 처리 노드
    EVALUATE_RESULT = "result_evaluator"  # 결과 평가 노드
    SUMMARIZE_RESULT = "result_summarizer"  # 결과 요약 노드


# SQL 코드 추출 유틸리티
def extract_sql_code(response: str) -> str:
    """AI 응답에서 SQL 코드 블록 추출"""
    if not response:
        return ""
    match = re.search(r"```sql\s*(.*?)\s*```", response.strip(), re.DOTALL)
    return match.group(1).strip() if match else ""


# 결과 제한 유틸리티 함수
def limit_result_for_llm(
    results: List[Any],
    max_rows: int = 20,
    max_str_length: int = 50,
    max_total_length: int = 3000,
) -> str:
    """
    SQL 결과를 LLM에 전달하기 위해 크기 제한

    Args:
        results: SQL 쿼리 결과 (첫 행은 헤더로 가정)
        max_rows: 최대 행 수 제한 (헤더 제외)
        max_str_length: 각 셀의 최대 문자열 길이
        max_total_length: 전체 결과 문자열 최대 길이

    Returns:
        제한된 결과를 포함한 문자열
    """
    if not isinstance(results, list) or len(results) <= 1:
        return "결과 없음"

    # 행/열 정보 계산
    total_rows = len(results) - 1  # 첫 번째 행은 컬럼명

    # 결과 제한
    if total_rows > max_rows:
        print(f"[SQL_GRAPH] 큰 결과 제한: {total_rows}행 -> {max_rows}행")
        limited_results = [results[0]] + results[
            1 : max_rows + 1
        ]  # 헤더 + 최대 max_rows개 행
        result_str = f"[결과 일부 - 전체 {total_rows}행 중 처음 {max_rows}행만 표시]\n"
    else:
        print(f"[SQL_GRAPH] 결과 전체 사용: {total_rows}행")
        limited_results = results
        result_str = ""

    # 결과 데이터를 압축된 형식으로 변환
    headers = limited_results[0]
    formatted_data = []

    for i in range(1, len(limited_results)):
        row_data = {}
        for j in range(min(len(headers), len(limited_results[i]))):
            col_name = str(headers[j])
            value = limited_results[i][j]
            # 긴 값 자르기
            if isinstance(value, str) and len(str(value)) > max_str_length:
                value = str(value)[:max_str_length] + "..."
            row_data[col_name] = value
        formatted_data.append(row_data)

    # 압축된 형식으로 결과 문자열 구성
    result_str += f"헤더: {headers}\n\n데이터 요약:\n"
    for i, row in enumerate(formatted_data):
        result_str += f"행 {i+1}: {row}\n"

    # 결과 문자열이 너무 길면 자르기
    if len(result_str) > max_total_length:
        result_str = result_str[: max_total_length - 3] + "..."

    return result_str


# 프롬프트 템플릿
class Prompts:
    # SQL 생성 프롬프트
    SQL_GENERATION = """
당신은 전문 SQL 생성기입니다. 
사용자는 한국어로 질문할 것이며, 당신은 유효한 PostgreSQL SQL 쿼리를 출력해야 합니다. 
주어진 스키마 정보를 사용하고 스키마에 없는 테이블/컬럼은 가정하지 마세요.

아래 규칙을 반드시 지켜주세요:
1. 마크다운 코드 블록(```sql)을 사용해서 원시 SQL 구문만 출력하세요.
2. 원시 SQL 구문만 출력하세요.
3. 서브 쿼리가 필요한 경우, 각각의 단계를 명확히 하여 sql 주석에 추가 해 주세요.

질문이 모호하거나 더 많은 정보가 필요하면 사용자에게 명확하게 해달라고 요청하세요.
"""

    # 결과 요약 프롬프트
    RESULT_SUMMARY = """
당신은 SQL 결과 분석 전문가입니다.
주어진 SQL 쿼리 결과를 분석하여 사용자의 원래 질문에 답변하는 간결한 요약을 생성하세요.
5문장 이내로 핵심만 답변하세요.
"""


# 그래프 노드 처리 클래스
class SQLGraphNodes:
    @staticmethod
    def generate_sql(state: SQLState) -> SQLState:
        """SQL 생성: 사용자 질문으로 SQL 쿼리 생성"""
        print(f"[SQL_GRAPH] SQL 생성 시작 - 질문: {state.question[:50]}...")

        # 상태 변수 설정
        retry_count = state.retry_count
        is_retry = retry_count > 0

        # 첫 실행 시 처리
        if not is_retry:
            # 원본 질문 저장
            state.original_question = state.question

            # 사용자 질문을 히스토리에 항상 추가 (중복 방지)
            history = state.history.copy()
            # # 마지막 사용자 메시지 확인
            # last_user = next(
            #     (m for m in reversed(history) if m["role"] == "user"), None
            # )

            # # 중복되지 않은 경우에만 추가 (비어있어도 추가)
            # if not last_user or last_user["content"] != state.question:
            history.append({"role": "user", "content": state.question})
            state.history = history

        # 프롬프트 설정 (항상 같은 프롬프트 사용)
        system_prompt = Prompts.SQL_GENERATION

        # 스키마/컨텍스트 정보 추가 (첫 실행 시만)
        if not is_retry:
            if state.schema:
                system_prompt += f"\n\n스키마 정보:\n{state.schema}"
            if state.context:
                system_prompt += f"\n\n추가 컨텍스트 정보:\n{state.context}"

        # 메시지 구성
        messages: List[BaseMessage] = [SystemMessage(content=system_prompt)]

        # 히스토리 추가
        for msg in state.history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

        # 현재 질문 추가
        messages.append(HumanMessage(content=state.question))

        # LLM 호출
        llm = ChatOpenAI(model=state.model, seed=0, temperature=0)
        response = llm.invoke(messages)
        content = str(response.content) if response.content else ""

        # SQL 코드 추출
        sql_code = extract_sql_code(content)
        print(f"[SQL_GRAPH] SQL {'추출 성공' if sql_code else '추출 실패'}")

        # 결과 설정 및 반환
        state.sql = sql_code
        state.error = None
        state.result = None
        state.llm_response = content

        return state

    @staticmethod
    def execute_sql(state: SQLState) -> SQLState:
        """SQL 실행: 생성된 쿼리 실행"""
        print("[SQL_GRAPH] SQL 실행 시작")

        sql = state.sql.strip()

        # 실행 조건 미충족 시
        if not sql:
            print("[SQL_GRAPH] SQL 쿼리 없음")
            state.result = None
            state.error = None
            return state

        retry_count = state.retry_count
        print(f"[SQL_GRAPH] SQL 실행 (시도 {retry_count+1})")

        # DB 설정이 없으면 기본값 사용
        db = state.db if state.db is not None else DBConfig()

        results, error = execute_query(db, sql)

        # 결과 로깅
        if error:
            print(f"[SQL_GRAPH] 오류: {error}")
        else:
            row_count = len(results) - 1 if results else 0
            print(f"[SQL_GRAPH] 결과: {row_count}개 행")

        state.result = results
        state.error = error
        return state

    @staticmethod
    def update_history(state: SQLState) -> SQLState:
        """히스토리 업데이트: 대화 기록 관리"""
        print("[SQL_GRAPH] 히스토리 업데이트")

        # 상태 정보
        sql = state.sql.strip()
        error = state.error
        results = state.result
        llm_response = state.llm_response
        summary = state.summary
        result_meta = state.result_meta
        history = state.history.copy()

        # LLM 응답이 있고 SQL이 없는 경우, 원본 응답 사용
        if llm_response and not sql:
            history.append({"role": "assistant", "content": llm_response})
            print("[SQL_GRAPH] SQL 없음, LLM 원본 응답 사용")

            # 질문 업데이트
            if not error and state.original_question:
                state.question = state.original_question

            state.history = history

            return state

        # SQL이 있는 경우 응답 구성
        parts = []
        if sql:
            parts.append(f"```sql\n{sql}\n```")

        if results:
            # 요약 결과가 있으면 사용
            if summary:
                parts.append(summary)
                if result_meta:
                    parts.append(result_meta)
                parts.append("자세한 결과는 '결과' 탭에서 확인하세요.")
            elif isinstance(results, list) and len(results) > 1:
                rows = len(results) - 1  # 첫 번째 행은 컬럼명
                cols = len(results[0]) if results and results[0] else 0
                parts.append(
                    f"쿼리 실행 결과: {rows}개의 행, {cols}개의 열이 조회되었습니다."
                )
                parts.append("자세한 결과는 '결과' 탭에서 확인하세요.")
            elif isinstance(results, list) and len(results) == 1:
                parts.append("쿼리가 성공적으로 실행되었지만 반환된 결과가 없습니다.")
            else:
                parts.append("쿼리가 성공적으로 실행되었습니다.")

        # 응답 추가
        content = "\n\n".join(parts)
        if content:
            history.append({"role": "assistant", "content": content})

        print(f"[SQL_GRAPH] 히스토리 업데이트 완료")

        # 히스토리 업데이트
        state.history = history

        # 질문 업데이트
        if not error and state.original_question:
            state.question = state.original_question

        return state

    @staticmethod
    def handle_error(state: SQLState) -> SQLState:
        """오류 처리: 에러 처리 및 재시도 관리"""
        error_msg = state.error or "알 수 없는 오류"
        print(f"[SQL_GRAPH] 오류 처리 - 오류: {error_msg}")

        # 재시도 카운터 증가
        retry_count = state.retry_count
        new_retry_count = retry_count + 1
        print(f"[SQL_GRAPH] 재시도 횟수: {retry_count} -> {new_retry_count}")

        # 변수 가져오기
        sql_query = state.sql.strip()
        original_q = state.original_question or state.question
        history = state.history.copy()

        # SQL 응답에 오류 정보 추가
        if sql_query:
            last_msg = history[-1] if history else {"role": "", "content": ""}
            sql_already_added = (
                last_msg["role"] == "assistant"
                and f"```sql\n{sql_query}" in last_msg["content"]
            )

            if not sql_already_added:
                history.append(
                    {
                        "role": "assistant",
                        "content": f"```sql\n{sql_query}\n```\n\n오류: {error_msg}",
                    }
                )

        # 오류 수정 요청을 유저 메시지로 추가
        history.append({"role": "user", "content": f"오류: {error_msg}"})

        # 오류 수정용 질문 생성 (간결하게)
        error_question = f"""
SQL 오류: {error_msg}

수정된 SQL 쿼리를 제공해주세요.
"""

        # 상태 업데이트
        state.retry_count = new_retry_count
        state.history = history
        state.original_question = original_q
        state.question = error_question

        return state

    @staticmethod
    def evaluate_result(state: SQLState) -> SQLState:
        """결과 평가: 쿼리 결과가 원본 질문에 충분히 답변하는지 평가"""
        # 상태 변수 확인
        original_question = state.original_question or ""
        results = state.result
        refine_count = state.refine_count
        sql = state.sql
        history = state.history.copy()

        print(f"[SQL_GRAPH] 결과 평가 (시도 {refine_count+1}/3)")

        # 평가 필요한 경우 체크:
        # 1. 결과가 있고 오류가 없는 경우
        # 2. 최대 시도 횟수(3회) 이내인 경우
        if not results or state.error or refine_count >= 2:
            print(
                f"[SQL_GRAPH] 평가 생략: 결과 없음/오류 발생/최대 시도 횟수({refine_count}/2) 초과"
            )
            return state

        # 평가 프롬프트 생성
        system_prompt = """
당신은 SQL 데이터 분석 전문가입니다. 
주어진 질문과 SQL 쿼리 결과를 분석하여, 결과가 질문에 충분히 답변하는지 평가해주세요.
결과가 부족하다면 'IMPROVE'를, 충분하다면 'SUFFICIENT'를 반환해주세요.
"""

        # 결과를 제한된 형식으로 변환
        result_info = limit_result_for_llm(results, max_rows=5, max_total_length=2000)

        # 메시지 구성
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=f"""
원본 질문: {original_question}

SQL 쿼리: 
```sql
{sql}
```

쿼리 결과: {result_info}

이 결과가 원본 질문에 충분히 답변하나요? 'IMPROVE' 또는 'SUFFICIENT'로 답변해주세요.
"""
            ),
        ]

        # LLM 호출
        llm = ChatOpenAI(model=state.model, seed=0, temperature=0)
        response = llm.invoke(messages)
        content = response.content
        evaluation = str(content).strip() if content is not None else ""

        # 평가 결과에 따른 처리
        if "IMPROVE" in evaluation.upper() and refine_count < 2:
            # 개선이 필요한 경우: 개선 요청 메시지 생성
            new_refine_count = refine_count + 1
            print(
                f"[SQL_GRAPH] 결과 개선 필요: 재생성 진행 (refine_count: {refine_count} -> {new_refine_count})"
            )

            # 히스토리에 결과와 개선 요청 추가
            if results:
                # 결과 정보 크기 제한
                if isinstance(results, list) and len(results) > 1:
                    rows = len(results) - 1
                    cols = len(results[0]) if results and results[0] else 0
                    result_info = (
                        f"쿼리 결과: {rows}개의 행, {cols}개의 열이 조회되었습니다."
                    )
                else:
                    result_info = "쿼리가 성공적으로 실행되었습니다."
                history.append({"role": "assistant", "content": result_info})

            # 개선 요청 질문 생성 - 데이터 크기 제한
            improve_question = f"""
현재 SQL 쿼리 결과가 다음 질문에 충분히 답변하지 못합니다:
질문: {original_question}

현재 쿼리:
```sql
{sql}
```

개선된 SQL 쿼리를 제공해주세요.
"""

            # 상태 업데이트 - refine_count 증가
            state.refine_count = new_refine_count
            state.history = history
            state.question = improve_question

            # 디버깅용 출력
            print(
                f"[SQL_GRAPH] refine_count 업데이트됨: {refine_count} -> {state.refine_count}"
            )
            return state
        else:
            # 결과가 충분한 경우 또는 최대 개선 시도 횟수 도달
            if "IMPROVE" in evaluation.upper():
                print(
                    f"[SQL_GRAPH] 개선이 필요하지만 최대 시도 횟수({refine_count}/2) 도달"
                )
            else:
                print("[SQL_GRAPH] 결과 충분함")
            return state

    @staticmethod
    def summarize_result(state: SQLState) -> SQLState:
        """결과 요약: SQL 쿼리 결과를 LLM으로 요약"""
        print("[SQL_GRAPH] 결과 요약 시작")

        # 결과 없는 경우 처리
        results = state.result
        sql = state.sql.strip()

        if not results or state.error:
            print("[SQL_GRAPH] 요약 생략: 결과 없음 또는 오류 발생")
            state.summary = None
            return state

        # 결과 데이터 확인
        if not isinstance(results, list) or len(results) <= 1:
            # 결과가 없거나 헤더만 있는 경우
            summary = "쿼리가 성공적으로 실행되었지만 반환된 결과가 없습니다."
            state.summary = summary
            return state

        # 행/열 정보 계산
        rows = len(results) - 1  # 첫 번째 행은 컬럼명
        cols = len(results[0]) if results and results[0] else 0
        meta_info = f"(쿼리 실행 결과: {rows}개의 행, {cols}개의 열)"

        # 기본 요약 메시지 (LLM 호출 실패 시 사용)
        basic_summary = (
            f"쿼리 실행 결과: {rows}개의 행, {cols}개의 열이 조회되었습니다."
        )

        # 결과 데이터 제한 및 변환
        result_str = limit_result_for_llm(results, max_rows=20, max_total_length=3000)

        # LLM 요약 시도
        summary = basic_summary
        try:
            original_question = state.original_question or state.question

            messages = [
                SystemMessage(content=Prompts.RESULT_SUMMARY),
                HumanMessage(
                    content=f"""
원본 질문: {original_question}

SQL 쿼리: 
```sql
{sql}
```

쿼리 결과: {result_str}

이 결과를 분석하고 사용자의 질문에 간결하게 답변하세요.
"""
                ),
            ]

            # LLM 호출
            llm = ChatOpenAI(model=state.model, seed=0, temperature=0)
            response = llm.invoke(messages)
            llm_summary = str(response.content) if response.content is not None else ""

            # LLM 요약이 있으면 사용
            if llm_summary.strip():
                summary = llm_summary
                print(f"[SQL_GRAPH] LLM 요약 성공")

        except Exception as e:
            # 오류 발생 시 기본 메시지 사용
            print(f"[SQL_GRAPH] LLM 요약 오류: {str(e)}")

        # 요약 결과 설정
        state.summary = summary
        state.result_meta = meta_info

        return state


# 조건부 분기 처리 클래스
class SQLGraphConditions:
    @staticmethod
    def should_retry(state: SQLState) -> str:
        """오류 발생 시 재시도 여부 결정"""
        retry_count = state.retry_count
        has_error = state.error

        # 재시도 최대 횟수는 5회
        if has_error and retry_count < 5:
            print(f"[SQL_GRAPH] 오류 발생, 재시도 진행 (retry_count: {retry_count}/5)")
            return "RETRY"
        if has_error:
            print(f"[SQL_GRAPH] 오류 발생, 최대 재시도 횟수({retry_count}/5) 초과")
            return "MAX_RETRY"

        print("[SQL_GRAPH] 오류 없음, 정상 처리")
        return "SUCCESS"

    @staticmethod
    def has_error(state: SQLState) -> str:
        """에러 여부에 따라 분기 결정"""
        has_error = bool(state.error)
        print(f"[SQL_GRAPH] {'오류 발생' if has_error else '오류 없음'}")
        return "HAS_ERROR" if has_error else "NO_ERROR"

    @staticmethod
    def has_sql(state: SQLState) -> str:
        """SQL 생성 여부에 따라 분기 결정"""
        sql = state.sql.strip()
        has_sql = bool(sql)
        print(f"[SQL_GRAPH] SQL {'생성됨' if has_sql else '생성 안됨'}")
        return "HAS_SQL" if has_sql else "NO_SQL"

    @staticmethod
    def should_refine(state: SQLState) -> str:
        """결과 평가 후 개선 여부 결정"""
        refine_count = state.refine_count
        original_question = state.original_question or ""

        # 최대 개선 시도 횟수(2회) 초과 시 항상 충분하다고 판단
        if refine_count >= 2:
            print(
                f"[SQL_GRAPH] 최대 시도 횟수({refine_count}/2) 도달, 더 이상 개선하지 않음"
            )
            return "SUFFICIENT"

        # refine_count가 증가했고 질문이 변경됐다면 개선이 필요함
        if state.question != original_question and refine_count > 0:
            print(
                f"[SQL_GRAPH] 결과 개선 필요: 질문 변경 및 refine_count={refine_count} 확인"
            )
            return "NEED_REFINE"

        print("[SQL_GRAPH] 결과 평가 완료: 충분함")
        return "SUFFICIENT"


def create_sql_graph():
    """SQL 쿼리 생성 및 실행 그래프 생성"""
    # TypedDict 대신 dataclass 사용
    graph = StateGraph(SQLState)

    # 노드 등록
    graph.add_node(Node.GENERATE_SQL, SQLGraphNodes.generate_sql)
    graph.add_node(Node.EXECUTE_SQL, SQLGraphNodes.execute_sql)
    graph.add_node(Node.UPDATE_HISTORY, SQLGraphNodes.update_history)
    graph.add_node(Node.HANDLE_ERROR, SQLGraphNodes.handle_error)
    graph.add_node(Node.EVALUATE_RESULT, SQLGraphNodes.evaluate_result)
    graph.add_node(Node.SUMMARIZE_RESULT, SQLGraphNodes.summarize_result)

    # SQL 생성 결과에 따른 분기
    graph.add_conditional_edges(
        Node.GENERATE_SQL,
        SQLGraphConditions.has_sql,
        {
            "HAS_SQL": Node.EXECUTE_SQL,  # SQL이 생성된 경우 실행
            "NO_SQL": Node.UPDATE_HISTORY,  # SQL이 생성되지 않은 경우 바로 히스토리 업데이트
        },
    )

    # 조건부 경로: 에러 처리
    graph.add_conditional_edges(
        Node.EXECUTE_SQL,
        SQLGraphConditions.has_error,
        {"HAS_ERROR": Node.HANDLE_ERROR, "NO_ERROR": Node.EVALUATE_RESULT},
    )

    # 조건부 경로: 재시도 처리
    graph.add_conditional_edges(
        Node.HANDLE_ERROR,
        SQLGraphConditions.should_retry,
        {
            "RETRY": Node.GENERATE_SQL,
            "MAX_RETRY": Node.UPDATE_HISTORY,
            "SUCCESS": Node.UPDATE_HISTORY,
        },
    )

    # 조건부 경로: 결과 평가 및 개선
    graph.add_conditional_edges(
        Node.EVALUATE_RESULT,
        SQLGraphConditions.should_refine,
        {
            "NEED_REFINE": Node.GENERATE_SQL,  # 개선 필요 시 SQL 재생성
            "SUFFICIENT": Node.SUMMARIZE_RESULT,  # 결과 충분 시 요약 노드로 이동
        },
    )

    # 결과 요약 후 히스토리 업데이트로 연결
    graph.add_edge(Node.SUMMARIZE_RESULT, Node.UPDATE_HISTORY)

    # 시작점과 종료점
    graph.set_entry_point(Node.GENERATE_SQL)
    graph.set_finish_point(Node.UPDATE_HISTORY)

    return graph.compile()


# 그래프 인스턴스 생성
sql_graph = create_sql_graph()
