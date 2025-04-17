"""
PostgreSQL 데이터베이스 연결 및 쿼리 실행을 위한 유틸리티 함수
"""

import psycopg
from psycopg import sql
from typing import List, Tuple, Dict, Any, Optional, Union

from text2sql.state import DBConfig


def connect_to_db(db: DBConfig) -> Optional[psycopg.Connection]:
    """
    PostgreSQL 데이터베이스에 연결합니다.

    Args:
        state: AppState 객체 (DB 설정을 가져옴)

    Returns:
        연결 객체 또는 연결 실패 시 None
    """
    try:
        # 상태에서 DB 설정 가져오기
        conn = psycopg.connect(
            host=db.host,
            port=db.port,
            dbname=db.dbname,
            user=db.user,
            password=db.password,
        )
        return conn
    except Exception as e:
        print(f"데이터베이스 연결 오류: {e}")
        return None


def execute_query(
    db: DBConfig,
    query: Union[str, sql.Composed],
    params: Optional[Union[Tuple, Dict[str, Any]]] = None,
) -> Tuple[Optional[List[Tuple]], Optional[str]]:
    """
    SQL 쿼리를 실행하고 결과를 반환합니다.

    Args:
        query: 실행할 SQL 쿼리 (문자열 또는 psycopg.sql.Composed 객체)
        params: 쿼리 파라미터 (선택 사항)
        state: AppState 객체 (DB 설정을 가져옴)

    Returns:
        (결과, 오류) 튜플:
            - 성공 시: (결과 리스트, None)
            - 실패 시: (None, 오류 메시지)
        SELECT 쿼리인 경우 첫 번째 행에 컬럼 이름이 포함됩니다.
    """
    conn = connect_to_db(db)
    if not conn:
        return None, "데이터베이스 연결에 실패했습니다."

    try:
        with conn.cursor() as cursor:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            # SELECT 쿼리인 경우 결과를 반환
            if cursor.description:
                # 컬럼 이름 정보 얻기
                column_names = [desc[0] for desc in cursor.description]
                # 데이터 결과 얻기
                results = cursor.fetchall()
                # 컬럼 이름을 첫 번째 행으로 추가
                return [tuple(column_names)] + results, None
            else:
                conn.commit()
                return [], None
    except Exception as e:
        return None, str(e)
    finally:
        if conn:
            conn.close()


def get_all_tables(db: DBConfig) -> Optional[List[str]]:
    """
    PostgreSQL 데이터베이스의 모든 테이블 목록을 반환합니다.

    Args:
        state: AppState 객체 (있는 경우 DB 포트를 가져옴)

    Returns:
        테이블 이름 목록 또는 실패 시 None
    """
    query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        ORDER BY table_name
    """

    results, error = execute_query(db, query)

    if error is None and results is not None:
        # 첫 번째 행은 컬럼 이름
        return [table[0] for table in results[1:]]
    return None


def get_table_schema(db: DBConfig, table_name: str) -> Optional[List[Tuple]]:
    """
    테이블의 스키마 정보(컬럼, 데이터 타입, NULL 허용 여부)를 반환합니다.

    Args:
        table_name: 스키마 정보를 조회할 테이블 이름
        state: AppState 객체 (있는 경우 DB 포트를 가져옴)

    Returns:
        컬럼 정보 목록 또는 실패 시 None
    """
    query = """
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = %s
        ORDER BY ordinal_position
    """

    results, error = execute_query(db, query, (table_name,))

    if error is None and results is not None:
        # 첫 번째 행은 컬럼 이름을 제외하고 반환
        return results[1:] if len(results) > 1 else []
    return None


def get_table_data(
    db: DBConfig, table_name: str, limit: int = 100
) -> Optional[List[Tuple]]:
    """
    테이블 데이터를 조회합니다.

    Args:
        table_name: 데이터를 조회할 테이블 이름
        limit: 반환할 최대 행 수
        state: AppState 객체 (있는 경우 DB 포트를 가져옴)

    Returns:
        테이블 데이터 또는 실패 시 None
    """
    query = sql.SQL("SELECT * FROM {} LIMIT %s").format(sql.Identifier(table_name))

    results, error = execute_query(db, query, (limit,))

    if error is None and results is not None:
        return results
    return None
