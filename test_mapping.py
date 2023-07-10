import pandas as pd

mapping = {'account_id': {
    "alias": "Account ID",
    "description": "Identifier of an account",
}, 'order_count_with_promo_category_0': {
    "alias": "Order Count (All Using Promo)= 0",
    "description": "Number of order from all categories that are using promo",
},
       'order_count_with_promo_category_1': {
           "alias": "Order Count (All Using Promo)= 1",
           "description": "Number of order from all categories that are using promo",
       },
       'order_count_with_promo_category_> 1': {
           "alias": "Order Count (All Using Promo)> 1",
           "description": "Number of order from all categories that are using promo",
       }, 'price_amount_category_0-280': {
           "alias": "Total Transaction Amount (All) <= 280",
           "description": "Total transaction amount from all categories" ,
       },
       'price_amount_category_281-870': {
           "alias": "Total Transaction Amount (All) 280-870",
           "description": "Total transaction amount from all categories",
       }, 'price_amount_category_871-2775': {
           "alias": "Total Transaction Amount (All) 871-2775",
           "description": "Total transaction amount from all categories",
       },
       'price_amount_category_> 2775': {
           "alias": "Total Transaction Amount (All) > 2775",
           "description": "Total transaction amount from all categories",
       }, 'promo_amount_category_0-16': {
           "alias": "Total Promocode Amount (All using Promo) < 16",
           "description": "Total promocode amount from all transactions that are using promo",
       },
       'promo_amount_category_16-81': {
           "alias": "Total Promocode Amount (All using Promo) 16-81",
           "description": "Total promocode amount from all transactions that are using promo",
       }, 'promo_amount_category_> 81': {
           "alias": "Total Promocode Amount (All using Promo) > 81",
           "description": "Total promocode amount from all transactions that are using promo",
       },
       'category_f_order_count_with_promo_category_0': {
           "alias": "Order Count (Category F using Promo) = 0",
           "description": "Number of order from category F that are using promo",
       },
       'category_f_order_count_with_promo_category_1': {
           "alias": "Order Count (Category F using Promo) = 1",
           "description": "Number of order from category F that are using promo",
       },
       'category_f_order_count_with_promo_category_2': {
           "alias": "Order Count (Category F using Promo) = 2",
           "description": "Number of order from category F that are using promo",
       },
       'category_f_order_count_with_promo_category_> 2': {
           "alias": "Order Count (Category F using Promo) > 2",
           "description": "Number of order from category F that are using promo",
       },
       'category_f_promo_amount_category_0-16': {
           "alias": "Total Promocode Amount (Category F using Promo) < 16",
           "description": "Total promocode amount from transactions in category F that are using promo",
       },
       'category_f_promo_amount_category_17-70': {
           "alias": "Total Promocode Amount (Category F using Promo) < 17-70",
           "description": "Total promocode amount from transactions in category F that are using promo",
       },
       'category_f_promo_amount_category_> 70': {
           "alias": "Total Promocode Amount (Category F using Promo) > 70",
           "description": "Total promocode amount from transactions in category F that are using promo",
       }, 'similar_email_category_0': {
           "alias": "Similar Email Count = 0",
           "description": "Number of account with similar email (similarity > 0.9)",
       },
       'similar_email_category_1': {
           "alias": "Similar Email Count = 1",
           "description": "Number of account with similar email (similarity > 0.9)",
       }, 'similar_email_category_2': {
           "alias": "Similar Email Count = 2",
           "description": "Number of account with similar email (similarity > 0.9)",
       },
       'similar_email_category_3': {
           "alias": "Similar Email Count = 3",
           "description": "Number of account with similar email (similarity > 0.9)",
       }, 'similar_email_category_4': {
           "alias": "Similar Email Count = 4",
           "description": "Number of account with similar email (similarity > 0.9)",
       },
       'similar_email_category_5': {
           "alias": "Similar Email Count = 5",
           "description": "Number of account with similar email (similarity > 0.9)",
       }, 'similar_email_category_> 5': {
           "alias": "Similar Email Count > 5",
           "description": "Number of account with similar email (similarity > 0.9)",
       },
       'similar_device_category_0': {
           "alias": "Similar Device Count = 0",
           "description": "Number of account with same device identifier",
       }, 'similar_device_category_1': {
           "alias": "Similar Device Count = 1",
           "description": "Number of account with same device identifier",
       },
       'similar_device_category_2': {
           "alias": "Similar Device Count = 2",
           "description": "Number of account with same device identifier",
       }, 'similar_device_category_> 2': {
           "alias": "Similar Device Count > 2",
           "description": "Number of account with same device identifier",
       }, 
       'label' : {
           "alias": "label",
           "description": "label"
       }
}

def get_alias(mapping, column_name):
    return mapping.get(column_name,"").get("alias", "")

def get_description(mapping, column_name):
    return mapping.get(column_name,"").get("description", "")


print(get_description(mapping ,"similar_email_category_2"))
# data_test = pd.read_csv("dataset/fix_data_test.csv")
# print(len(data_test))
# print(data_test.columns)