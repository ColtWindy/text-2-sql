다음은 데이터베이스 테이블정의야. 같은 라인 주석에 테이블의 의미가 정의 되어 있어.

- `customers` 테이블
  - `customer_id`: 주문 당 하나씩 생성 됨. order 테이블을 조회 할 때에는 customer_id를 사용해야 해.
  - `customer_unique_id`: 고객의 고유 id. 이며 테이블내 중복존재 가능 하므로, 고객을 구분하기 위해서 사용 되어야 해.

```sql
CREATE TABLE public.customers ( -- public.customers
cuid text, -- customer_id text
cuidq text, -- customer_unique_id text
czcp bigint, -- customer_zip_code_prefix bigint
ccty text, -- customer_city text
cste text -- customer_state text
);

ALTER TABLE public.customers OWNER TO postgres;

CREATE TABLE public.geolocation (
gzcp bigint, -- geolocation_zip_code_prefix bigint
glat real, -- geolocation_lat real
glng real, -- geolocation_lng real
gcty text, -- geolocation_city text
gste text -- geolocation_state text
);

ALTER TABLE public.geolocation OWNER TO postgres;

CREATE TABLE public.leads_closed (
mqid text, -- mql_id text
slid text, -- seller_id text
sdr text, -- sdr_id text
srid text, -- sr_id text
wdat text, -- won_date text
bsmt text, -- business_segment text
ltyp text, -- lead_type text
lbpf text, -- lead_behaviour_profile text
hcmp bigint, -- has_company bigint
hgtin bigint, -- has_gtin bigint
avst text, -- average_stock text
btyp text, -- business_type text
dpcs real, -- declared_product_catalog_size real
dmrv real -- declared_monthly_revenue real
);

ALTER TABLE public.leads_closed OWNER TO postgres;

CREATE TABLE public.leads_qualified (
mqid text, -- mql_id text
dmrv text, -- first_contact_date text
lpid text, -- landing_page_id text
origin text -- origin text
);

ALTER TABLE public.leads_qualified OWNER TO postgres;

CREATE TABLE public.order_items (
orid text, -- orid text
oitd bigint, -- oitd bigint
prid text, -- prid text
slid text, -- slid text
sldt text, -- sldt text
pric real, -- pric real
frtv real -- frtv real
);

ALTER TABLE public.order_items OWNER TO postgres;

CREATE TABLE public.order_payments (
orid text, -- order_id text
pymt bigint, -- payment_sequential bigint
pyty text, -- payment_type text
pyin bigint, -- payment_installments bigint
pyvl real -- payment_value real
);

ALTER TABLE public.order_payments OWNER TO postgres;

CREATE TABLE public.order_reviews (
rvid text, -- review_id text
orid text, -- order_id text
rvsc bigint, -- review_score bigint
rvct text, -- review_comment_title text
rvcmsg text, -- review_comment_message text
rvcdt text, -- review_creation_date text
rvat text -- review_answer_timestamp text
);

ALTER TABLE public.order_reviews OWNER TO postgres;

CREATE TABLE public.orders (
orid text, -- order_id text
cuid text, -- customer_id text
orst text, -- order_status text
opts text, -- order_purchase_timestamp text
opat text, -- order_approved_at text
odcd text, -- order_delivered_carrier_date text
odct text, -- order_delivered_customer_date text
oedt text -- order_estimated_delivery_date text
);

ALTER TABLE public.orders OWNER TO postgres;

CREATE TABLE public.product_category_name_translation ( -- public.product_category_name_translation
pcnt text, -- product_category_name text
pcnte text -- product_category_name_english text
);

ALTER TABLE public.product_category_name_translation OWNER TO postgres;

CREATE TABLE public.products (
prid text, -- product_id text
pcnt text, -- product_category_name text
pnlh real, -- product_name_lenght real
pdlh real, -- product_description_lenght real
ppqt real, -- product_photos_qty real
pwgt real, -- product_weight_g real
plcm real, -- product_length_cm real
phcm real, -- product_height_cm real
pwcm real -- product_width_cm real
);

ALTER TABLE public.products OWNER TO postgres;

CREATE TABLE public.sellers (
slid text, -- seller_id text
szcp bigint, -- seller_zip_code_prefix bigint
scty text, -- seller_city text
sste text -- seller_state text
);

ALTER TABLE public.sellers OWNER TO postgres;
```
