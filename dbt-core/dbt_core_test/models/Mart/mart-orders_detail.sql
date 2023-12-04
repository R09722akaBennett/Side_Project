{{ config(materialized='table') }}

with cte as (select
*
from {{ref('stg-orders')}} o
left join {{ref('stg-items_records')}}  subtotal
using (order_number)),

full_data as (
select
cte.order_number,
cte.status,
cte.customer_id,
cte.customer_name,
cte.customer_phone,
cte.created_from,
cte.created_at,
cte.product_subscription_period,
cte.subtotal_dollars,
cte.order_discount_dollars,
cte.user_credit_dollars,
cte.total_dollars,
cte.order_points_to_cash_dollars,
cte.invoice_buyer_name,
cte.agent_name,
cte.created_by_channel_name,
cte.delivery_address_country,
cte.delivery_address_city,
cte.delivery_address_state,
cte.utm_source,
cte.utm_medium,
cte.utm_campaign,
cte.utm_term,
cte.membership_tier_name,
item.item_type,
item.item_id,
item.item_name,
item.sku,
item.quantity,
item.item_price,
item.total_dollars as item_total_dollars,
item.created_by,
from cte , unnest(items) as item
)
select
*
from full_data