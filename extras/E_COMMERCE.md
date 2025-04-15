## 데이터 베이스 설명

- `customers` 테이블
  - `customer_id`: 주문 당 하나씩 생성 됨. order 테이블을 조회 할 때에는 customer_id를 사용해야 해.
  - `customer_unique_id`: 고객의 고유 id. 이며 테이블내 중복존재 가능 하므로, 고객을 구분하기 위해서 사용 되어야 해

## 스키마

```sql
-- DROP SCHEMA public;

CREATE SCHEMA public AUTHORIZATION postgres;
-- public.customers definition

CREATE TABLE public.customers (
customer_id text NULL,
customer_unique_id text NULL,
customer_zip_code_prefix int8 NULL,
customer_city text NULL,
customer_state text NULL
);

-- public.geolocation definition

CREATE TABLE public.geolocation (
geolocation_zip_code_prefix int8 NULL,
geolocation_lat float4 NULL,
geolocation_lng float4 NULL,
geolocation_city text NULL,
geolocation_state text NULL
);

-- public.leads_closed definition

CREATE TABLE public.leads_closed (
mql_id text NULL,
seller_id text NULL,
sdr_id text NULL,
sr_id text NULL,
won_date text NULL,
business_segment text NULL,
lead_type text NULL,
lead_behaviour_profile text NULL,
has_company int8 NULL,
has_gtin int8 NULL,
average_stock text NULL,
business_type text NULL,
declared_product_catalog_size float4 NULL,
declared_monthly_revenue float4 NULL
);

-- public.leads_qualified definition

CREATE TABLE public.leads_qualified (
mql_id text NULL,
first_contact_date text NULL,
landing_page_id text NULL,
origin text NULL
);

-- public.order_items definition

CREATE TABLE public.order_items (
order_id text NULL,
order_item_id int8 NULL,
product_id text NULL,
seller_id text NULL,
shipping_limit_date text NULL,
price float4 NULL,
freight_value float4 NULL
);

-- public.order_payments definition

CREATE TABLE public.order_payments (
order_id text NULL,
payment_sequential int8 NULL,
payment_type text NULL,
payment_installments int8 NULL,
payment_value float4 NULL
);

-- public.order_reviews definition

CREATE TABLE public.order_reviews (
review_id text NULL,
order_id text NULL,
review_score int8 NULL,
review_comment_title text NULL,
review_comment_message text NULL,
review_creation_date text NULL,
review_answer_timestamp text NULL
);

-- public.orders definition

CREATE TABLE public.orders (
order_id text NULL,
customer_id text NULL,
order_status text NULL,
order_purchase_timestamp text NULL,
order_approved_at text NULL,
order_delivered_carrier_date text NULL,
order_delivered_customer_date text NULL,
order_estimated_delivery_date text NULL
);

-- public.product_category_name_translation definition

CREATE TABLE public.product_category_name_translation (
product_category_name text NULL,
product_category_name_english text NULL
);

-- public.products definition

CREATE TABLE public.products (
product_id text NULL,
product_category_name text NULL,
product_name_lenght float4 NULL,
product_description_lenght float4 NULL,
product_photos_qty float4 NULL,
product_weight_g float4 NULL,
product_length_cm float4 NULL,
product_height_cm float4 NULL,
product_width_cm float4 NULL
);

-- public.sellers definition

CREATE TABLE public.sellers (
seller_id text NULL,
seller_zip_code_prefix int8 NULL,
seller_city text NULL,
seller_state text NULL
);
```
