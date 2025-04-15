## local 003

```json
{
  "instance_id": "local003",
  "db": "E_commerce",
  "question": "According to the RFM definition document, calculate the average sales per order for each customer within distinct RFM segments, considering only 'delivered' orders. Use the customer unique identifier. Clearly define how to calculate Recency based on the latest purchase timestamp and specify the criteria for classifying RFM segments. The average sales should be computed as the total spend divided by the total number of orders. Please analyze and report the differences in average sales across the RFM segments",
  "external_knowledge": "RFM.md"
},
```


```sql
WITH
RecencyScore AS (
    SELECT
        customer_unique_id,
        last_purchase,
        NTILE(5) OVER (ORDER BY last_purchase DESC) AS recency
    FROM (
        SELECT
            customer_unique_id,
            MAX(order_purchase_timestamp) AS last_purchase
        FROM orders
        JOIN customers USING (customer_id)
        WHERE order_status = 'delivered'
        GROUP BY customer_unique_id
    ) AS sub
),
FrequencyScore AS (
    SELECT
        customer_unique_id,
        total_orders,
        NTILE(5) OVER (ORDER BY total_orders DESC) AS frequency
    FROM (
        SELECT
            customer_unique_id,
            COUNT(order_id) AS total_orders
        FROM orders
        JOIN customers USING (customer_id)
        WHERE order_status = 'delivered'
        GROUP BY customer_unique_id
    ) AS sub
),
MonetaryScore AS (
    SELECT
        customer_unique_id,
        total_spent,
        NTILE(5) OVER (ORDER BY total_spent DESC) AS monetary
    FROM (
        SELECT
            customer_unique_id,
            SUM(price) AS total_spent
        FROM orders
        JOIN order_items USING (order_id)
        JOIN customers USING (customer_id)
        WHERE order_status = 'delivered'
        GROUP BY customer_unique_id
    ) AS sub
),

RFM AS (
    SELECT
        r.customer_unique_id,
        r.last_purchase,
        f.total_orders,
        m.total_spent,
        CASE
            WHEN r.recency = 1
                 AND (f.frequency + m.monetary) IN (1, 2, 3, 4) THEN 'Champions'

            WHEN r.recency IN (4, 5)
                 AND (f.frequency + m.monetary) IN (1, 2) THEN 'Can''t Lose Them'

            WHEN r.recency IN (4, 5)
                 AND (f.frequency + m.monetary) IN (3, 4, 5, 6) THEN 'Hibernating'

            WHEN r.recency IN (4, 5)
                 AND (f.frequency + m.monetary) IN (7, 8, 9, 10) THEN 'Lost'

            WHEN r.recency IN (2, 3)
                 AND (f.frequency + m.monetary) IN (1, 2, 3, 4) THEN 'Loyal Customers'

            WHEN r.recency = 3
                 AND (f.frequency + m.monetary) IN (5, 6) THEN 'Needs Attention'

            WHEN r.recency = 1
                 AND (f.frequency + m.monetary) IN (7, 8) THEN 'Recent Users'

            WHEN (r.recency = 1 AND (f.frequency + m.monetary) IN (5, 6))
                 OR  (r.recency = 2 AND (f.frequency + m.monetary) IN (5, 6, 7, 8))
            THEN 'Potentital Loyalists'

            WHEN r.recency = 1
                 AND (f.frequency + m.monetary) IN (9, 10) THEN 'Price Sensitive'

            WHEN r.recency = 2
                 AND (f.frequency + m.monetary) IN (9, 10) THEN 'Promising'

            WHEN r.recency = 3
                 AND (f.frequency + m.monetary) IN (7, 8, 9, 10) THEN 'About to Sleep'
        END AS rfm_bucket
    FROM RecencyScore AS r
    JOIN FrequencyScore AS f USING (customer_unique_id)
    JOIN MonetaryScore AS m USING (customer_unique_id)
)

SELECT
    rfm_bucket,
    AVG(total_spent / total_orders) AS avg_sales_per_customer
FROM RFM
GROUP BY rfm_bucket
ORDER BY rfm_bucket;
```

```csv
rfm_bucket,avg_sales_per_customer
About to Sleep,56.82698235193433
Can't Lose Them,351.0409347428705
Champions,246.78421133129848
Hibernating,182.88007343622496
Lost,56.9497533180623
Loyal Customers,242.69821088912923
Needs Attention,144.52755916145654
Potentital Loyalists,130.5285709305824
Price Sensitive,34.92501948233797
Promising,34.484931325978216
Recent Users,66.87964358130482
```

## local 004

```json
{
  "instance_id": "local004",
  "db": "E_commerce",
  "question": "Could you tell me the number of orders, average payment per order and customer lifespan in weeks of the 3 custumers with the highest average payment per order, where the lifespan is calculated by subtracting the earliest purchase date from the latest purchase date in days, dividing by seven, and if the result is less than seven days, setting it to 1.0?",
  "external_knowledge": null
}
```

```sql
WITH CustomerData AS (
    SELECT
        customer_unique_id,
        COUNT(DISTINCT orders.order_id) AS order_count,
        SUM(payment_value) AS total_payment,
        MIN(order_purchase_timestamp)::date AS first_order_day,
        MAX(order_purchase_timestamp)::date AS last_order_day
    FROM customers
        JOIN orders USING (customer_id)
        JOIN order_payments USING (order_id)
    GROUP BY customer_unique_id
)
SELECT
    customer_unique_id,
    order_count AS PF,
    ROUND((total_payment / order_count)::numeric, 2) AS AOV,
    CASE
        WHEN (last_order_day - first_order_day) < 7 THEN
            1
        ELSE
            (last_order_day - first_order_day) / 7
    END AS ACL
FROM CustomerData
ORDER BY AOV DESC
LIMIT 3;
```
