import pandas as pd

df = pd.read_csv("tests/export (4).csv")

df_fraud = df[df["label"] == "fraud"]

print(df.columns)
# Get feature columns and label
df_feature = df_fraud.iloc[:,-6:]
print(df_feature.head())

agg_df = pd.DataFrame()
feature_columns = df_feature.columns
for col in feature_columns:
    agg_df[col] = df_feature[col].apply(lambda x: 1 if x != 0 else 0)

# print(agg_df.head())
print("AGG DF")
# print(agg_df.head())
total_length = len(agg_df)

percentages = pd.DataFrame()
for col in agg_df.columns:
    if col != 'label':
        values_greater_than_1 = sum(1 for value in agg_df[col] if value > 0)
        percentage = (values_greater_than_1 / total_length) * 100
        print(col,":",percentage)

        percentages = percentages.append({'Columns': col, 'Percentages': percentage}, ignore_index=True)


        # percentages[col] = percentage
sorted_percentages = percentages.sort_values(by='Percentages', ascending=False)

print("sorted",sorted_percentages)

print(sorted_percentages.to_dict(orient="records"))

