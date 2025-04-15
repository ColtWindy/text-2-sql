
---
## 동기
- 쿼리작성하는 고통
	- 레거시
	- 이해가 어려운 컬럼명
	- 종종 코드 레벨까지...
- 협업
	- 간단한 쿼리라도 효율이 떨어진다 -> SQL 배우세요
	- 통계 뽑아주세요.
- CS
	- 과거 데이터 조회 -> 백오피스, ADMIN

---



---
## 데이터셋
---
- Spider 2
- Kaggle: [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
![[HRhd2Y0 1.png]]


---
**customer_id, customer_unique_id**: A customer has a customer_id per order, whereas he gets only one customer_unique_id in the database)

예제 데이터 설명


---

Spider 2에 해당 테스트는 두 개 (`local003`, `local004`) 존재함. 이 중 어려운 난이도로 테스트
### `local003`
- [Instruction](https://github.com/xlang-ai/Spider2/blob/b1dbe41e544409f2f6e72652cd45be8e00bca176/spider2/examples/spider2.jsonl#L608)
- [SQL](https://github.com/xlang-ai/Spider2/blob/b1dbe41e544409f2f6e72652cd45be8e00bca176/spider2-lite/evaluation_suite/gold/sql/local003.sql#L4)
#### Instruction
> 배송 완료의 데이터만을 대상으로, 고객(customer) 각각의 R(recency score)을 구해줘, (1 은 최근 구매, 5 는 오래전 구매). 최신 구매 시각(order_purchase_timestamp)을 사용해서 최근순으로 정렬해 주고. 고객의 고유 ID(unique id)를 사용해줘. 구매시간을 포함해서 알려줘


---
## 애플리케이션 구현

- Streamlit
- Langgraph
- Pandas
---

## Application, Agent

- 쿼리 분기
- 오류 수정
- 쿼리 평가
- 결과 취합(summarize)
- 정확도
- 모델(GPT4o)
---
## Graph 애플리케이션 구현

Agent 구성
![[Pasted image 20250415080450.png]]

---
## Agent 구성 요소
### State
```python
class SQLState(TypedDict):
    question: str  # 사용자 질문
    schema: str  # DB 스키마 정보
    context: str  # 추가 컨텍스트 정보
    sql: str  # 생성된 SQL 쿼리
    error: Optional[str]  # SQL 실행 에러
    result: Any  # SQL 실행 결과
    history: List[dict]  # 대화 히스토리
    model: str  # 사용할 모델명
    llm_response: Optional[str]  # LLM 응답
    retry_count: int  # 재시도 횟수
    original_question: str  # 원본 질문
    refine_count: int  # 결과 평가 횟수
    summary: Optional[str]  # 결과 요약
    result_meta: Optional[str]  # 결과 메타 정보
```

### Nodes
```python
    GENERATE_SQL = "sql_generator"  # SQL 생성 노드
    EXECUTE_SQL = "sql_executor"  # SQL 실행 노드
    UPDATE_HISTORY = "history_updater"  # 히스토리 업데이트 노드
    HANDLE_ERROR = "error_handler"  # 에러 처리 노드
    EVALUATE_RESULT = "result_evaluator"  # 결과 평가 노드
    SUMMARIZE_RESULT = "result_summarizer"  # 결과 요약 노드
```

> LangGraph Studio를 참고하세요!

---

## 시스템 요청을 무시하는 질문
SQL만 답하라는 시스템 프롬프트를 무시하고 LLM이 그 외 답변을 하는 경우

### gpt-4o: 텍스트와 sql을 모두 작성하는 프롬프트
- 가장 많은 매출이 발생한 사람은 누구지?


### SQL 만 발생시키는 질문
- 몇 명이 주문 했을까?
	- 고객에 대한 구분은 uid로 구분해야 해

### 역으로 물어보는 질문
- 몇 명이야?

```sql
SELECT COUNT(*) FROM orders;
```


```sql
SELECT COUNT(DISTINCT customer_id) FROM orders
```

이 경우는 `customer_id`를 고유하다고 판단하기 때문에, `customer_id`는 주문의 갯수만큼 존재한다.

- 가장 많이 물건을 판 셀러는 누구지?

```sql
SELECT seller_id, COUNT(*) AS total_items_sold
FROM order_items
GROUP BY seller_id
ORDER BY total_items_sold DESC
LIMIT 1;
```

## 요청 난이도에 따른 분류

### Spider 2 local003
-  RFM의 스코어를 각각 계산해줘,  최근 구매 날짜를 기준으로 해 줘. 계산된 스코어를 기반으로 각각 세그먼트를 나누고 세그먼트 이름과 함께, 세그먼트 내의 평균 매출(sales) 금액을 함께 알려줘. 고객 구분은 unique id로 해 줘

> RFM 문서 참고: RFM은 마케팅에서 자주 사용하는 용어예요. Recency(최근성), Frequency(빈도), Monetary(구매 금액)의 약자로, 고객의 구매 행동을 기반으로 고객을 세분화하는 지표입니다. 이 스코어를 계산해 고객을 등급별로 분류하고, 맞춤형 마케팅 전략을 세우는 데 활용하죠. 예를 들면, 최근에 자주, 많이 구매한 고객을 우수 고객으로 분류해 특별 혜택을 제공하는 방식이에요.

### 난이도 1: 기본적인 SELECT 및 WHERE
- 배송 완료된 주문은 몇 개인가?
- 등록된 판매자는 총 몇 명인가
- 고객들이 거주하는 서로 다른 주는 어디인가?
- 상파울루 시에 있는 판매자의 ID와 주를 모두 알려줘.
- 신용카드로 결제된 주문 건수는 몇 건인가?

### 난이도 2: JOIN 및 기본 집계/그룹화
- 각 주문의 고객 고유 ID를 알려줘. 
- 각 상품의 카테고리 이름을 보여줘. (`JOIN`)
- 주문 상태별 주문 건수를 알려줘. (`COUNT`)
- 신용카드로 2회 이상 할부 결제된 주문의 ID와 결제 금액을 보여줘. (`WHERE`)
- 리뷰 평점이 5점인 주문들의 주문 상태를 비교해 줘 (`COUNT, JOIN, WHERE, GROUP BY`)

### 난이도 3: GROUP BY + HAVING, 서브쿼리, 날짜 조건 처리
- 두 개 이상의 아이템을 포함한 주문 ID를 모두 보여줘. (`HAVING`)
- 예상 배송일보다 늦게 배송된 주문을 찾아줘.
- 2017년 월별 주문 건수를 알려줘.
- 가장 높은 운송료를 지불한 주문의 ID를 알려줘.(서브쿼리를 사용해줘)

### 난이도 4: 다중 JOIN, 고급 집계 및 서브쿼리/CTE 활용
- 두 명 이상의 판매자가 포함된 주문을 찾아줘. (`GROUP BY, HAVING, COUNT`)
- *상품 카테고리별 평균 상품 가격과 평균 운송료를 알려줘.* ( 다중 `JOIN, GROUP BY, AVG` >2. 영어 이름으로 해 줘)
- 2018년 월별 총 판매금액을 계산해줘.
- 주문 건수가 가장 많은 상위 5명의 판매자의 ID와 주문 건수를 보여줘. (> cte 사용해줘)

### 난이도 5: 윈도우 함수, 복잡한 조인/서브쿼리 및 고급 기능 활용
- 주(state)별 주문 건수와 해당 순위를 보여줘. (`CTE, JOIN, GROUP BY, COUNT, RANK(), ...)
```sql
WITH state_order_counts AS (
    SELECT
        c.customer_state,
        COUNT(o.order_id) AS order_count
    FROM
        customers c
    JOIN
        orders o ON c.customer_id = o.customer_id
    GROUP BY
        c.customer_state
)

SELECT
    customer_state,
    order_count,
    RANK() OVER (ORDER BY order_count DESC) AS rank
FROM
    state_order_counts
ORDER BY
    rank;
```
- *각 상품 카테고리별로 가장 무거운 커테고리의 ID와 무게를 알려줘.*
	- 해석이 달라질 수 있음. 카테고리 ID 일 수도, 무게일수도 있음. > 카테고리를 기준으로 하는 경우 *각 상품 카테고리별로 가장 무거운 커테고리의 ID와 무게를 알려줘.*
```sql
SELECT product_category_name, product_id, product_weight_g 
FROM (
  SELECT product_category_name, product_id, product_weight_g,
         ROW_NUMBER() OVER (
           PARTITION BY product_category_name 
           ORDER BY product_weight_g DESC
         ) AS rn 
  FROM olist_products_dataset
) AS sub
WHERE rn = 1;
```
- 2회 이상 주문한 고객 수를 알려줘.
- *평균 리뷰 평점이 가장 높은 상품 카테고리와 그 평점을 구해줘.*
```sql
SELECT t.product_category_name_english, 
       AVG(r.review_score) AS avg_score 
FROM olist_order_reviews_dataset AS r 
JOIN olist_orders_dataset AS o 
  ON r.order_id = o.order_id 
JOIN olist_order_items_dataset AS oi 
  ON o.order_id = oi.order_id 
JOIN olist_products_dataset AS p 
  ON oi.product_id = p.product_id 
JOIN product_category_name_translation AS t 
  ON p.product_category_name = t.product_category_name 
GROUP BY t.product_category_name_english 
ORDER BY avg_score DESC 
LIMIT 1;
```
- *날짜별 총 판매금액과 누적 판매금액을 계산해줘.*
```sql
SELECT order_date, 
       daily_sales, 
       SUM(daily_sales) OVER (ORDER BY order_date) AS cumulative_sales 
FROM (
  SELECT DATE(order_purchase_timestamp) AS order_date, 
         SUM(oi.price) AS daily_sales 
  FROM olist_orders_dataset AS o 
  JOIN olist_order_items_dataset AS oi 
    ON o.order_id = oi.order_id 
  GROUP BY DATE(order_purchase_timestamp)
) AS sub
ORDER BY order_date;
```



## Agent 작성시 참고

- IN (...) 과 JOIN (...) 을 사용하는 차이는 explain등을 통해서 분석하고 더 나은 방식을 사용한다.

## 모델별로 다른 행동
- 매출순으로 10명을 보여줘
	- gpt-4o > 시스템 프롬프트를 무시하고 한글과 함께 출력
```sql
SELECT c.customer_id, SUM(oi.price) AS total_revenue
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
JOIN order_items oi ON o.order_id = oi.order_id
GROUP BY c.customer_id
ORDER BY total_revenue DESC
LIMIT 10;
```
	
	- gpt-4.5-preview-2025-02-27 > 정상적으로 출력

```sql
SELECT c.customer_id, SUM(op.payment_value) AS total_sales
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
JOIN order_payments op ON o.order_id = op.order_id
GROUP BY c.customer_id
ORDER BY total_sales DESC
LIMIT 10;
```

## 컨텍스트에 대한 주의 사항

### 새로운 대화를 시작하고 바로 다음을 입력하면? LLM은 어떻게 해석 할까?

- 매출순으로 10명을 보여줘
---
## 배운점
- 쓸만하다.
- 쿼리 요청은 해석의 여지가 있다. 정확성과 편의성의 균형이 필요할 듯.
- 복잡한 쿼리는 말로 표현하기도 어렵다. > 적절히 테이블 정보나 로직을 던져주는게 실용적
- LangGraph 의 흐름은 명확한게 좋다.
- SEED
---
## TODO

### 메타데이터 업데이트
- 파일 편집이 가능한 에이전트
- 에이턴트를 통해 실시간으로 채팅을 통해 편집하고, 이를 쿼리에 즉각 반영하는 형태

### Explain 결과 LLM
- 쿼리 최적화
- 실행 전략 및 DB리소스 관리

### 테이블 추출(Pruning)
- 스키마 정보 자체가 많은 경우, 컨텍스트 제한
- 쿼리를 위해 필요한 테이블 추출 방법 (최소한의 데이터를 점진적으로 반복, 분할 정복?)
### 보안
- 키워드 검증 (CREATE, DELETE, DROP, ALTER )
- LLM을 활용

---
#### 보안주의!

어느 샌가! 자동으로 테이블이 생성 되었음. (score를 계산하라는 걸 저장하라는 것으로 이해 한 듯?)
![[Pasted image 20250414232519.png]]
