WITH recency_scores AS (
    SELECT 
        c.customer_unique_id,
        NTILE(5) OVER (ORDER BY MAX(o.order_purchase_timestamp) DESC) AS recency_score
    FROM 
        orders o
    JOIN 
        customers c ON o.customer_id = c.customer_id
    WHERE 
        o.order_status = 'delivered'
    GROUP BY 
        c.customer_unique_id
),
frequency_scores AS (
    SELECT 
        c.customer_unique_id,
        NTILE(5) OVER (ORDER BY COUNT(o.order_id) DESC) AS frequency_score
    FROM 
        orders o
    JOIN 
        customers c ON o.customer_id = c.customer_id
    WHERE 
        o.order_status = 'delivered'
    GROUP BY 
        c.customer_unique_id
),
monetary_scores AS (
    SELECT 
        c.customer_unique_id,
        NTILE(5) OVER (ORDER BY SUM(oi.price) DESC) AS monetary_score
    FROM 
        orders o
    JOIN 
        order_items oi ON o.order_id = oi.order_id
    JOIN 
        customers c ON o.customer_id = c.customer_id
    WHERE 
        o.order_status = 'delivered'
    GROUP BY 
        c.customer_unique_id
),
rfm_scores AS (
    SELECT 
        r.customer_unique_id,
        r.recency_score,
        f.frequency_score,
        m.monetary_score,
        (f.frequency_score + m.monetary_score) AS fm_score
    FROM 
        recency_scores r
    JOIN 
        frequency_scores f ON r.customer_unique_id = f.customer_unique_id
    JOIN 
        monetary_scores m ON r.customer_unique_id = m.customer_unique_id
),
segments AS (
    SELECT 
        customer_unique_id,
        CASE
            WHEN recency_score = 1 AND fm_score BETWEEN 1 AND 4 THEN 'Champions'
            WHEN recency_score IN (4, 5) AND fm_score BETWEEN 1 AND 2 THEN 'Cant Lose Them'
            WHEN recency_score IN (4, 5) AND fm_score BETWEEN 3 AND 6 THEN 'Hibernating'
            WHEN recency_score IN (4, 5) AND fm_score BETWEEN 7 AND 10 THEN 'Lost'
            WHEN recency_score IN (2, 3) AND fm_score BETWEEN 1 AND 4 THEN 'Loyal Customers'
            WHEN recency_score = 3 AND fm_score BETWEEN 5 AND 6 THEN 'Needs Attention'
            WHEN recency_score = 1 AND fm_score BETWEEN 7 AND 8 THEN 'Recent Users'
            WHEN (recency_score = 1 AND fm_score BETWEEN 5 AND 6) OR (recency_score = 2 AND fm_score BETWEEN 5 AND 8) THEN 'Potential Loyalists'
            WHEN recency_score = 1 AND fm_score BETWEEN 9 AND 10 THEN 'Price Sensitive'
            WHEN recency_score = 2 AND fm_score BETWEEN 9 AND 10 THEN 'Promising'
            WHEN recency_score = 3 AND fm_score BETWEEN 7 AND 10 THEN 'About to Sleep'
            ELSE 'Other'
        END AS segment
    FROM 
        rfm_scores
),
average_revenue AS (
    SELECT 
        s.segment,
        AVG(oi.price) AS average_revenue
    FROM 
        segments s
    JOIN 
        orders o ON s.customer_unique_id = (SELECT c.customer_unique_id FROM customers c WHERE c.customer_id = o.customer_id)
    JOIN 
        order_items oi ON o.order_id = oi.order_id
    WHERE 
        o.order_status = 'delivered'
    GROUP BY 
        s.segment
)
SELECT 
    segment,
    average_revenue
FROM 
    average_revenue;