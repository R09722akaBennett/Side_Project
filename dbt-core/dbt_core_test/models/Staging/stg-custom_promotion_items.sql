{{ config(materialized='table') }}

with custom_discount_item as(

SELECT 
JSON_EXTRACT_scalar(_airbyte_data, "$.id") as order_id,
JSON_EXTRACT_scalar(_airbyte_data, "$.order_number") as order_number,
---------------------------------------------------------------------
JSON_EXTRACT_scalar(custom_discount_items, "$.id") as id, --subtotal_item.item_dat.order_custom_discount_items.order_discounted_price
JSON_EXTRACT_scalar(custom_discount_items, "$.item_type") as item_type,
-- JSON_EXTRACT_scalar(custom_discount_items, "$.item_data.discount_special_flag") as discount_special_flag,
JSON_EXTRACT_scalar(custom_discount_items, "$.item_data.name") as item_data_name,
-- JSON_EXTRACT_scalar(custom_discount_items, "$.item_data.discount_type") as item_data_discount_type,
-- JSON_EXTRACT_scalar(custom_discount_items, "$.item_data.discount_level") as item_data_discount_level,
-- JSON_EXTRACT_scalar(custom_discount_items, "$.item_data.discount_id") as item_data_discount_id,
JSON_EXTRACT_scalar(custom_discount_items, "$.item_id") as item_id,
JSON_EXTRACT_scalar(custom_discount_items, "$.total.dollars") as total_dollars,
 FROM `norse-quest-379015.source.source` ,  
 unnest(JSON_EXTRACT_array(_airbyte_data, "$.custom_discount_items") ) as custom_discount_items
  )

select 
order_number,
array_agg(
  struct(
    id,
    item_type,
    custom_discount_item.item_data_name,
    item_id,
    total_dollars
  )
)custom_discount_item
from custom_discount_item
group by order_number