{{ config(materialized='table') }}

with orders_detail as(

SELECT 
JSON_EXTRACT_scalar(_airbyte_data, "$.id") as order_id,
JSON_EXTRACT_scalar(_airbyte_data, "$.order_number") as order_number,
---------------------------------------------------------------------
JSON_EXTRACT_scalar(subtotal_items, "$.id") as id,
JSON_EXTRACT_scalar(subtotal_items, "$.item_type") as item_type,
JSON_EXTRACT_scalar(subtotal_items, "$.item_data.cart_item_id") as cart_item_id,
JSON_EXTRACT_scalar(subtotal_items, "$.item_data.promotion_id") as promotion_id,
JSON_EXTRACT_scalar(subtotal_items, "$.item_data.parent_item_ids[0]") as parent_item_ids,
JSON_EXTRACT_scalar(subtotal_items, "$.item_data.triggering_item_id") as triggering_item_id,
JSON_EXTRACT_scalar(subtotal_items, "$.item_data.order_promotion_items.discounted_price") as discounted_price,
JSON_EXTRACT_scalar(subtotal_items, "$.item_data.order_custom_discount_items.discounted_price") as order_custom_discount_items_discounted_price,
JSON_EXTRACT_scalar(subtotal_items, "$.item_data.has_exclude_promotion_tag") as has_exclude_promotion_tag,
JSON_EXTRACT_scalar(subtotal_items, "$.item_id") as item_id,
JSON_EXTRACT_scalar(subtotal_items, "$.item_price.dollars") as item_price,
JSON_EXTRACT_scalar(subtotal_items, "$.media._id") as media_id,
JSON_EXTRACT_scalar(subtotal_items, "$.media.images.original.url") as media_url,
JSON_EXTRACT_scalar(subtotal_items, "$.product_subscription_id") as product_subscription_id,
JSON_EXTRACT_scalar(subtotal_items, "$.title_translations.zh-hant") as item_name,
JSON_EXTRACT_scalar(subtotal_items, "$.sku") as sku,
JSON_EXTRACT_scalar(subtotal_items, "$.quantity") as quantity,
JSON_EXTRACT_scalar(subtotal_items, "$.total.dollars") as total_dollars,
JSON_EXTRACT_scalar(subtotal_items, "$.order_discounted_price.dollars") as order_discounted_price_dollars,
JSON_EXTRACT_scalar(subtotal_items, "$.created_by") as created_by,

 FROM `norse-quest-379015.source.source` , 
 unnest(JSON_EXTRACT_array(_airbyte_data, "$.subtotal_items") ) as subtotal_items
  )

select 
order_number,
array_agg(
  struct(
    id,
    item_type,
    item_id,
    item_name,
    sku,
    quantity,
    item_price,
    total_dollars,
    order_discounted_price_dollars,
    order_custom_discount_items_discounted_price,
    discounted_price,
    created_by
  )) items
from orders_detail
group by order_number
