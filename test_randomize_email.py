from operator import index
import pandas as pd

def concat_dataframes(df1, df2):
    # Get the index of the "account_id" column in the first DataFrame
    account_id_index = df1.columns.get_loc("account_id")
    
    # Insert the "email" column from the second DataFrame after the "account_id" column
    df1.insert(account_id_index + 1, "email", df2["email"])
    
    return df1

data_test = pd.read_csv("dataset/fix_data_test.csv")
print("Data test",len(data_test))
data_val = pd.read_csv("dataset/fix_data_val.csv")
print("Data val",len(data_val))
final_train = pd.read_csv("dataset/fix_data_train.csv")
# print(data_train["label"].value_counts())
# print("Data train",len(data_train))
# # Check for duplicate values in the "account_id" column
# duplicates = data_val.duplicated("account_id")
# # Get the duplicated rows based on the "account_id" column
# duplicated_rows = data_val[duplicates]
# print("Duplicates",duplicated_rows)




email_test = pd.read_csv("dataset/fix_email_test.csv")
print(len(email_test))
email_val = pd.read_csv("dataset/fix_email_val.csv")
print(len(email_val))
# email_train = pd.read_csv("dataset/fix_email_train.csv")
# print(len(email_train))
    
final_test = concat_dataframes(data_test, email_test)
print(final_test.head())
final_val = concat_dataframes(data_val, email_val)
print(final_val.head())
# final_train = concat_dataframes(data_train, email_train)
# print(final_train.head())

final_test.to_csv("dataset/final_data_test.csv",index=False)
final_val.to_csv("dataset/final_data_val.csv", index=False)
final_train.to_csv("dataset/final_data_train.csv", index=False)