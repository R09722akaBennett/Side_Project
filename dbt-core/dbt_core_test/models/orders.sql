
/*
    Welcome to your first dbt model!
    Did you know that you can also configure models directly within SQL files?
    This will override configurations stated in dbt_project.yml

    Try changing "table" to "view" below
*/

{{ config(materialized='table') }}

with source_data as (

select
JSON_EXTRACT_scalar(_airbyte_data, "$.id") as id,
JSON_EXTRACT_scalar(_airbyte_data, "$.order_number") as order_number,
JSON_EXTRACT_scalar(_airbyte_data, "$.status") as status,
JSON_EXTRACT_scalar(_airbyte_data, "$.order_remarks") as order_remarks,
JSON_EXTRACT_scalar(_airbyte_data, "$.customer_id") as customer_id,
JSON_EXTRACT_scalar(_airbyte_data, "$.customer_name") as customer_name,
JSON_EXTRACT_scalar(_airbyte_data, "$.customer_phone") as customer_phone,
JSON_EXTRACT_scalar(_airbyte_data, "$.created_from") as created_from,
JSON_EXTRACT_scalar(_airbyte_data, "$.updated_at") as updated_at,
JSON_EXTRACT_scalar(_airbyte_data, "$.created_at") as created_at,
JSON_EXTRACT_scalar(_airbyte_data, "$.ga_tracked") as ga_tracked,
JSON_EXTRACT_scalar(_airbyte_data, "$.is_guest_checkout") as is_guest_checkout,
JSON_EXTRACT_scalar(_airbyte_data, "$.default_warehouse_id") as default_warehouse_id,
JSON_EXTRACT_scalar(_airbyte_data, "$.return_from_order_id") as default_warehouse_name,
JSON_EXTRACT_scalar(_airbyte_data, "$.default_warehouse_address") as default_warehouse_address,
JSON_EXTRACT_scalar(_airbyte_data, "$.product_subscription_period") as product_subscription_period,
JSON_EXTRACT_scalar(_airbyte_data, "$.order_payment.id") as order_payment_id,
JSON_EXTRACT_scalar(_airbyte_data, "$.order_payment.payment_method.id") as order_payment_method_id,
JSON_EXTRACT_scalar(_airbyte_data, "$.order_payment.payment_type") as payment_type,
JSON_EXTRACT_scalar(_airbyte_data, "$.order_payment.name_translations.zh-cn") as payment_name,
JSON_EXTRACT_scalar(_airbyte_data, "$.order_payment.name_translations.zh-hant") as payment_name_hant,
JSON_EXTRACT_scalar(_airbyte_data, "$.order_payment.status") as payment_status,
JSON_EXTRACT_scalar(_airbyte_data, "$.order_payment.total_dollars") as payment_amount,
JSON_EXTRACT_scalar(_airbyte_data, "$.order_payment.ref_payment_id") as ref_payment_id,
JSON_EXTRACT_scalar(_airbyte_data, "$.subtotal.dollars") as subtotal_dollars,
JSON_EXTRACT_scalar(_airbyte_data, "$.order_discount.dollars") as order_discount_dollars,
JSON_EXTRACT_scalar(_airbyte_data, "$.user_credit.dollars") as user_credit_dollars,
JSON_EXTRACT_scalar(_airbyte_data, "$.total.dollars") as total_dollars,
JSON_EXTRACT_scalar(_airbyte_data, "$.order_points_to_cash.dollars") as order_points_to_cash_dollars,
JSON_EXTRACT_scalar(_airbyte_data, "$.invoice.tax_id") as invoice_tax_id,
JSON_EXTRACT_scalar(_airbyte_data, "$.invoice.buyer_name") as invoice_buyer_name,
JSON_EXTRACT_scalar(_airbyte_data, "$.invoice.mailling_address") as invoice_mailling_address,
JSON_EXTRACT_scalar(_airbyte_data, "$.invoice.shipping_address") as invoice_shipping_address,
JSON_EXTRACT_scalar(_airbyte_data, "$.channel.created_by_channel_id") as created_by_channel_id,
JSON_EXTRACT_scalar(_airbyte_data, "$.channel.created_by_channel_name.zh-hant") as created_by_channel_name,
JSON_EXTRACT_scalar(_airbyte_data, "$.agent.agent_name") as agent_name,
JSON_EXTRACT_scalar(_airbyte_data, "$.order_source.id") as order_source_id,
JSON_EXTRACT_scalar(_airbyte_data, "$.order_source.type") as order_source_type,
JSON_EXTRACT_scalar(_airbyte_data, "$.order_source.source_id") as order_source_source_id,
JSON_EXTRACT_scalar(_airbyte_data, "$.order_source.name.zh-hant") as order_source_name, 
JSON_EXTRACT_scalar(_airbyte_data, "$.delivery_address.country") as delivery_address_country,
JSON_EXTRACT_scalar(_airbyte_data, "$.delivery_address.city") as delivery_address_city,
JSON_EXTRACT_scalar(_airbyte_data, "$.delivery_address.state") as delivery_address_state,
JSON_EXTRACT_scalar(_airbyte_data, "$.delivery_address.postcode") as delivery_address_postcode,
JSON_EXTRACT_scalar(_airbyte_data, "$.delivery_address.address_1") as delivery_address_1,
JSON_EXTRACT_scalar(_airbyte_data, "$.delivery_address.address_2") as delivery_address_2,
JSON_EXTRACT_scalar(_airbyte_data, "$.delivery_address.logistic_codes") as delivery_address_logistic_codes,
JSON_EXTRACT_scalar(_airbyte_data, "$.delivery_address.name") as delivery_address_name,
JSON_EXTRACT_scalar(_airbyte_data, "$.delivery_address.phone") as delivery_address_phone,
JSON_EXTRACT_scalar(_airbyte_data, "$.utm_data.utm_source") as utm_source,
JSON_EXTRACT_scalar(_airbyte_data, "$.utm_data.utm_medium") as utm_medium,
JSON_EXTRACT_scalar(_airbyte_data, "$.utm_data.utm_campaign") as utm_campaign,
JSON_EXTRACT_scalar(_airbyte_data, "$.utm_data.utm_term") as utm_term,
JSON_EXTRACT_scalar(_airbyte_data, "$.utm_data.utm_time") as utm_time,
JSON_EXTRACT_scalar(_airbyte_data, "$.membership_tier_data.name") as membership_tier_name,

from `norse-quest-379015.source.source`

)

select *
from source_data

/*
    Uncomment the line below to remove records with null `id` values
*/

-- where id is not null
