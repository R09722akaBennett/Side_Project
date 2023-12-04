{{ config(materialized='table') }}

with promotion_items as(

SELECT 
JSON_EXTRACT_scalar(_airbyte_data, "$.id") as order_id,
JSON_EXTRACT_scalar(_airbyte_data, "$.order_number") as order_number,
---------------------------------------------------------------------
JSON_EXTRACT_scalar(promotion_items, "$.id") as id,
JSON_EXTRACT_scalar(promotion_items, "$.discounted_amount.dollars") as discounted_amount_dollars,
JSON_EXTRACT_scalar(promotion_items, "$.subtotal_after.dollars") as subtotal_after_dollars,
-- JSON_EXTRACT_scalar(promotion_items, "$.promotion._id") as promotion_id,
JSON_EXTRACT_STRING_ARRAY(promotion_items, "$.promotion.available_platforms") as available_platforms,
-- JSON_EXTRACT_scalar(promotion_items, "$.promotion.codes") as codes,
JSON_EXTRACT_scalar(promotion_items, "$.promotion.created_at") as promotion_created_at,
JSON_EXTRACT_scalar(promotion_items, "$.promotion.discount_on") as discount_on,
JSON_EXTRACT_scalar(promotion_items, "$.promotion.discount_percentage") as discount_percentage,
JSON_EXTRACT_scalar(promotion_items, "$.promotion.discount_type") as discount_type,
JSON_EXTRACT_STRING_ARRAY(promotion_items, "$.promotion.discountable_category_ids") as discountable_category_ids,
JSON_EXTRACT_STRING_ARRAY(promotion_items, "$.promotion.discountable_product_ids") as discountable_product_ids,
JSON_EXTRACT_scalar(promotion_items, "$.promotion.discountable_quantity") as discountable_quantity,
JSON_EXTRACT_scalar(promotion_items, "$.promotion.discounted_point") as discounted_point,
JSON_EXTRACT_scalar(promotion_items, "$.promotion.discounted_price") as discounted_price,
-- JSON_EXTRACT_scalar(promotion_items, "$.promotion.end_at") as end_at,
-- JSON_EXTRACT_scalar(promotion_items, "$.promotion.extended_promotion_id") as extended_promotion_id,
-- JSON_EXTRACT_scalar(promotion_items, "$.promotion.first_purchase_only") as first_purchase_only,
-- JSON_EXTRACT_scalar(promotion_items, "$.promotion.for_affiliate_campaign") as for_affiliate_campaign,
-- JSON_EXTRACT_scalar(promotion_items, "$.promotion.is_accumulated") as is_accumulated,
-- JSON_EXTRACT_scalar(promotion_items, "$.promotion.max_use_count") as max_use_count,
-- JSON_EXTRACT_scalar(promotion_items, "$.promotion.membership_tier_id") as membership_tier_id,
-- JSON_EXTRACT_scalar(promotion_items, "$.promotion.merchant_id") as merchant_id,
-- JSON_EXTRACT_scalar(promotion_items, "$.promotion.multiple_code") as multiple_code,
-- JSON_EXTRACT_scalar(promotion_items, "$.promotion.requires_membership") as requires_membership,
-- JSON_EXTRACT_scalar(promotion_items, "$.promotion.seo_keywords") as seo_keywords,
-- JSON_EXTRACT_scalar(promotion_items, "$.promotion.show_coupon") as show_coupon,
-- JSON_EXTRACT_scalar(promotion_items, "$.promotion.start_at") as start_at,
-- JSON_EXTRACT_scalar(promotion_items, "$.promotion.status") as status,
JSON_EXTRACT_scalar(promotion_items, "$.promotion.title_translations.zh-hant") as title_translations,
-- JSON_EXTRACT_scalar(promotion_items, "$.promotion.updated_at") as promotion_updated_at,
-- JSON_EXTRACT_scalar(promotion_items, "$.promotion.use_count") as use_count,
-- JSON_EXTRACT_scalar(promotion_items, "$.promotion.user_max_use_count") as user_max_use_count,
-- JSON_EXTRACT_scalar(promotion_items, "$.promotion.is_extend_promotion") as is_extend_promotion,
-- JSON_EXTRACT_scalar(promotion_items, "$.promotion.id") as promotion_id,
JSON_EXTRACT_scalar(promotion_items, "$.updated_at") as updated_at,
JSON_EXTRACT_scalar(promotion_items, "$.created_at") as created_at,
-- JSON_EXTRACT_array(promotion_items, "$.item_data") as item_data,
-- JSON_EXTRACT_scalar(promotion_items, "$.item_data.type") as item_data_type,
-- JSON_EXTRACT_scalar(promotion_items, "$.item_data.promotion_id") as item_data_promotion_id,
-- JSON_EXTRACT_scalar(promotion_items, "$.item_data.parent_item_ids") as item_data_parent_item_ids,
-- JSON_EXTRACT_scalar(promotion_items, "$.item_data.id") as item_data_id,
-- JSON_EXTRACT_array(promotion_items, "$.promotion_conditions_data") as promotion_conditions_data,
-- JSON_EXTRACT_scalar(promotion_items, "$.coupon_code") as coupon_code,

 FROM `norse-quest-379015.source.source` , 
 unnest(JSON_EXTRACT_array(_airbyte_data, "$.promotion_items") ) as promotion_items

  )

select 
order_number, 
array_agg(
  struct(
      id, -- subtotal_items.item_data.order_promotion_items.discounted_price
      discounted_amount_dollars,
      subtotal_after_dollars,
      available_platforms,
      promotion_created_at,
      discount_on,
      discount_percentage,
      discount_type,
      discountable_category_ids,
      discountable_product_ids,
      discountable_quantity,
      discounted_point,
      discounted_price,
      title_translations,
      updated_at,
      created_at
  )) promotion
from promotion_items 
group by order_number
