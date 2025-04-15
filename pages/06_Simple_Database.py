import streamlit as st
from text2sql.db_utils import get_all_tables, get_table_schema, get_table_data
import pandas as pd

st.title("데이터베이스 테이블 조회")

if st.button("모든 테이블 조회"):
    tables = get_all_tables()

    if tables:
        st.success(f"{len(tables)}개의 테이블을 찾았습니다.")

        # 테이블 목록 표시
        st.subheader("테이블 목록")
        for i, table in enumerate(tables, 1):
            st.write(f"{i}. {table}")

            # 테이블 구조 조회
            columns = get_table_schema(table)
            if columns:
                with st.expander(f"{table} 구조 상세보기"):
                    # 컬럼 정보를 데이터프레임으로 변환하여 표시
                    column_data = {
                        "컬럼명": [col[0] for col in columns],
                        "데이터 타입": [col[1] for col in columns],
                        "Nullable": [col[2] for col in columns],
                    }
                    st.dataframe(column_data)

                    # 미리보기 추가
                    data = get_table_data(table, limit=10)
                    if data and len(data) > 0:
                        st.write(f"{table} 데이터 미리보기 (최대 10개 행):")

                        # 컬럼명
                        column_names = [col[0] for col in columns]

                        # 데이터프레임으로 변환
                        df = pd.DataFrame(data, columns=column_names)
                        st.dataframe(df)
                    else:
                        st.info("테이블에 데이터가 없습니다.")
    else:
        st.warning("테이블을 찾을 수 없거나 데이터베이스 연결에 실패했습니다.")
