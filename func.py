
from fasterrisk import RiskScoreOptimizer, RiskScoreClassifier
from utils import download_file_from_google_drive,  compute_logisticLoss_from_X_y_beta0_betas, get_all_product_booleans, get_support_indices, isEqual_upTo_8decimal, isEqual_upTo_16decimal, get_all_product_booleans
import pickle
import re
# from sklearn.model_selection import train_test_split
# from imblearn import SMOTE

import pandas as pd
import numpy as np
import time

# TODO: Create mapping feature name and description

# === MAPPING === #
feature_info_mapping = {'account_id': {
    "alias": "Account ID",
    "description": "Identifier of an account",
}, 'order_count_with_promo_category_0': {
    "alias": "Order Count (All Using Promo) = 0",
    "description": "Number of order from all categories that are using promo is zero (no order)",
},
       'order_count_with_promo_category_1': {
           "alias": "Order Count (All Using Promo) = 1",
           "description": "Number of order from all categories that are using promo is 1",
       },
       'order_count_with_promo_category_> 1': {
           "alias": "Order Count (All Using Promo) > 1",
           "description": "Number of order from all categories that are using promo is more than 1",
       }, 'price_amount_category_0-280': {
           "alias": "Total Transaction Amount (All) <= 280",
           "description": "Total transaction amount from all categories is lower than 280" ,
       },
       'price_amount_category_281-870': {
           "alias": "Total Transaction Amount (All) 280-870",
           "description": "Total transaction amount from all categories is between 280 and 870",
       }, 'price_amount_category_871-2775': {
           "alias": "Total Transaction Amount (All) 871-2775",
           "description": "Total transaction amount from all categories is between 871 and 2775",
       },
       'price_amount_category_> 2775': {
           "alias": "Total Transaction Amount (All) > 2775",
           "description": "Total transaction amount from all categories is higher than 2775",
       }, 'promo_amount_category_0-16': {
           "alias": "Total Promocode Amount (All using Promo) < 16",
           "description": "Total promocode amount from all transactions that are using promo is lower than 16",
       },
       'promo_amount_category_16-81': {
           "alias": "Total Promocode Amount (All using Promo) 16-81",
           "description": "Total promocode amount from all transactions that are using promo is between 16 and 81",
       }, 'promo_amount_category_> 81': {
           "alias": "Total Promocode Amount (All using Promo) > 81",
           "description": "Total promocode amount from all transactions that are using promo is more than 81",
       },
       'category_f_order_count_with_promo_category_0': {
           "alias": "Order Count (Category F using Promo) = 0",
           "description": "Number of order from category F that are using promo is zero (no order)",
       },
       'category_f_order_count_with_promo_category_1': {
           "alias": "Order Count (Category F using Promo) = 1",
           "description": "Number of order from category F that are using promo is 1",
       },
       'category_f_order_count_with_promo_category_2': {
           "alias": "Order Count (Category F using Promo) = 2",
           "description": "Number of order from category F that are using promo is 2",
       },
       'category_f_order_count_with_promo_category_> 2': {
           "alias": "Order Count (Category F using Promo) > 2",
           "description": "Number of order from category F that are using promo is more than 2",
       },
       'category_f_promo_amount_category_0-16': {
           "alias": "Total Promocode Amount (Category F using Promo) < 16",
           "description": "Total promocode amount from transactions in category F that are using promo is less than 16",
       },
       'category_f_promo_amount_category_17-70': {
           "alias": "Total Promocode Amount (Category F using Promo) = 17-70",
           "description": "Total promocode amount from transactions in category F that are using promo is between 17 and 70",
       },
       'category_f_promo_amount_category_> 70': {
           "alias": "Total Promocode Amount (Category F using Promo) > 70",
           "description": "Total promocode amount from transactions in category F that are using promo is higher than 70",
       }, 'similar_email_category_0': {
           "alias": "Similar Email Count = 0",
           "description": "Number of account with similar email (similarity > 0.9) is 0",
       },
       'similar_email_category_1': {
           "alias": "Similar Email Count = 1",
           "description": "Number of account with similar email (similarity > 0.9) is 1",
       }, 'similar_email_category_2': {
           "alias": "Similar Email Count = 2",
           "description": "Number of account with similar email (similarity > 0.9) is 2",
       },
       'similar_email_category_3': {
           "alias": "Similar Email Count = 3",
           "description": "Number of account with similar email (similarity > 0.9) is 3",
       }, 'similar_email_category_4': {
           "alias": "Similar Email Count = 4",
           "description": "Number of account with similar email (similarity > 0.9) is 4",
       },
       'similar_email_category_5': {
           "alias": "Similar Email Count = 5",
           "description": "Number of account with similar email (similarity > 0.9) is 5",
       }, 'similar_email_category_> 5': {
           "alias": "Similar Email Count > 5",
           "description": "Number of account with similar email (similarity > 0.9) is more than 5",
       },
       'similar_device_category_0': {
           "alias": "Similar Device Count = 0",
           "description": "Number of account with same device identifier is 0",
       }, 'similar_device_category_1': {
           "alias": "Similar Device Count = 1",
           "description": "Number of account with same device identifier is 1",
       },
       'similar_device_category_2': {
           "alias": "Similar Device Count = 2",
           "description": "Number of account with same device identifier is 2",
       }, 'similar_device_category_> 2': {
           "alias": "Similar Device Count > 2",
           "description": "Number of account with same device identifier is more than 2",
       }, 
       'label' : {
           "alias": "label",
           "description": "label"
       }
}

def get_alias(column_name):
    return feature_info_mapping[column_name].get("alias", "")

def get_description(column_name):
    return feature_info_mapping[column_name].get("description", "")

# === DATA PREPROCESSING === #

def read_data(file_path):
    ''' Read data from file path and return dataframe '''
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df, target_col):
    ''' Preprocess dfframe and return numpy array X and y '''
    df[target_col] = df[target_col].map({1:1 ,0:-1, -1:-1})
    # Convert boolean columns to integer
    bool_columns = df.select_dtypes(include=['bool']).columns
    df[bool_columns] = df[bool_columns].astype(int)
    # Return as numpy
    numpy_data = np.asarray(df)
    X, y = numpy_data[:, :-1], numpy_data[:, -1]
    return X, y

# def drop_unused_columns(df):
#     if 'account_id' in df.columns:
#         df = df.drop(['account_id'], axis=1)
#     return df

# === TRAIN === #
def train_model(X_train, y_train, sparsity):
    # Initialize a risk score optimizer
    m = RiskScoreOptimizer(X = X_train, y = y_train, k = sparsity)
    # Perform optimization
    m.optimize()
    # # Get all top m solutions from the final diverse pool
    # arr_multiplier, arr_intercept, arr_coefficients = m.get_models() # get m solutions from the diverse pool; Specifically, arr_multiplier.shape=(m, ), arr_intercept.shape=(m, ), arr_coefficients.shape=(m, p)
    # get the first solution from the final diverse pool by passing an optional model_index; models are ranked in order of increasing logistic loss
    multiplier, intercept, coefficients = m.get_models(model_index = 0) # get the first solution (smallest logistic loss) from the diverse pool; Specifically, multiplier.shape=(1, ), intercept.shape=(1, ), coefficients.shape=(p, )
    # Get a classifier
    clf = RiskScoreClassifier(multiplier=multiplier, intercept=intercept, coefficients=coefficients)
    return clf

# === PREDICT === #
def get_prediction(model, X_test):
    y_pred = model.predict(X = X_test)
    return y_pred

# === GET DATA === #

def get_calculation_table(risk_score_model):
    assert risk_score_model.featureNames is not None, "please pass the featureNames to the model by using the function .reset_featureNames(featureNames)"

    nonzero_indices = get_support_indices(risk_score_model.coefficients)

    max_feature_length = max([len(featureName) for featureName in risk_score_model.featureNames])
    row_score_template = '{0}. {1:>%d}     {2:>2} point(s) | + ...' % (max_feature_length)

    print("The Risk Score is:")
    for count, feature_i in enumerate(nonzero_indices):
        row_score_str = row_score_template.format(count+1, risk_score_model.featureNames[feature_i], int(risk_score_model.coefficients[feature_i]))
        if count == 0:
            row_score_str = row_score_str.replace("+", " ")

        print(row_score_str)

    final_score_str = ' ' * (14+max_feature_length) + 'SCORE | =    '
    print(final_score_str)
    
    print("###")
    feature_names_list = []
    coefficients_list = []
    features_list = []
    for count, feature_i in enumerate(nonzero_indices):
        feature_name = risk_score_model.featureNames[feature_i]
        # feature_name = convert_string_format(risk_score_model.featureNames[feature_i])
        # feature_name = get_alias(risk_score_model.featureNames[feature_i])
        print("feature",feature_name)
        print("converted feature",convert_string_format(feature_name))
        feature_dict = { feature_name : int(risk_score_model.coefficients[feature_i]) }
        feature_names_list.append(feature_name)
        coefficients_list.append(int(risk_score_model.coefficients[feature_i]))
        features_list.append(feature_dict)
    
    feature_alias_list = [get_alias(x) for x in feature_names_list]
    feature_desc_list = [get_description(x) for x in feature_names_list]

    print("feature names: ", feature_names_list)
    print("coefficients: ", coefficients_list)
    print(len(feature_names_list) == len(coefficients_list))

    # Create a risk score mapping
    all_product_booleans = get_all_product_booleans(len(nonzero_indices))
    all_scores = all_product_booleans.dot(risk_score_model.coefficients[nonzero_indices])
    all_scores = np.unique(all_scores)
    all_scaled_scores = (risk_score_model.intercept + all_scores) / risk_score_model.multiplier
    all_risks = 1 / (1 + np.exp(-all_scaled_scores))
    scores_list = all_scores.tolist()
    risks_list = all_risks.tolist()
    # Map score (integer) to risk percentage
    risk_template = '{0:>5}%'

    mapping = {}
    for index, item_a in enumerate(scores_list):
        if index < len(risks_list):
            mapping[int(item_a)] = risk_template.format(round(100*risks_list[index], 1))
    print("Mapping risk score",mapping)

    return { "model": features_list, 
            "features": feature_names_list,
            "features_alias": feature_alias_list,
            "features_description": feature_desc_list,
            "mapping": mapping }

# === === #

# === Utils === #
def sort_data(data, sort_by, order):
    print(data[:19])
    sorted_data = sorted(data, key=lambda x: x.get(sort_by))
    if order == 'desc':
        sorted_data.reverse()
    return sorted_data

def search_data(prediction_list, search_query: str):
    filtered_list = []
    regex_pattern = re.compile(search_query, re.IGNORECASE)
    for prediction in prediction_list:
        # Check if any column value matches the search query using regex
        for value in prediction.values():
            if isinstance(value, str) and regex_pattern.search(value):
                filtered_list.append(prediction)
                break  # Break out of the inner loop if match found for any column
            elif isinstance(value, int) and regex_pattern.search(str(value)):
                filtered_list.append(prediction)
                break  # Break out of the inner loop if match found for any column
    return filtered_list

# === #
# def get_predictions_from_data(model, data):
#     cleaned_data = preprocess_data(data,"label")
#     print(cleaned_data)
#     predictions = get_prediction(model, cleaned_data)
#     print(predictions)
#     return predictions

def train_fasterrisk_with_smote(dataset_name):
    # Get training data
    file_path = "dataset/" + dataset_name + ".csv"
    train_data = read_data(file_path)
    print(train_data.head())
    if 'account_id' in train_data.columns:
        train_data = train_data.drop(['account_id'],axis=1)
    if 'email' in train_data.columns:
        train_data = train_data.drop(['email'],axis=1)
    print(train_data.head())
    print(len(train_data.columns), train_data.columns)
    X_train, y_train = preprocess_data(train_data, 'label')
    print(X_train.shape)
    print(y_train.shape)
    # print(X_train)
    # print(y_train)
    # Modeling
    sparsity = 5
    parent_size = 10
    # Create risk score optimizer
    RiskScoreOptimizer_m = RiskScoreOptimizer(X = X_train, y = y_train, k = sparsity, parent_size = parent_size)
    RiskScoreOptimizer_m.optimize()
    multipliers, sparseDiversePool_beta0_integer, sparseDiversePool_betas_integer = RiskScoreOptimizer_m.get_models()
   
    # Dump top 5 model based on logistic loss
    for i in range (5):
        # Select first model (best model based on logistic loss)
        model_index = i # first model
        multiplier = multipliers[model_index]
        intercept = sparseDiversePool_beta0_integer[model_index]
        coefficients = sparseDiversePool_betas_integer[model_index]
        model = RiskScoreClassifier(multiplier, intercept, coefficients)
        X_featureNames = list(train_data.columns)
        model.reset_featureNames(X_featureNames)
        # model.print_model_card()
        model_file_name = "model/model_{}.pkl".format(model_index + 1)
        # Dump object into a pickle file
        with open(model_file_name, "wb") as f:
            pickle.dump(model, f)
            print("Dumped model {}".format(model_index + 1))
    
    print("Finished dump 5 models")
def apply_mapping(df, mapping):
    for mapping_dict in mapping:
        for column, value in mapping_dict.items():
            df[column] = df[column].apply(lambda x: int(value) if x == 1 else int(x))
    return df

def calculate_scores(df, model_dict):
    ''' Function to add total score and risk percentage column'''
    data = model_dict['model']
    mapping = model_dict['mapping']
    print("Columns: ",df.columns)
    def calculate_score(row):
        score = 0
        for item in data:
            key = next(iter(item))
            if key in row.index:
                value = item[key]
                if row[key] > 0:
                # if value > 0 and row[key] > 0:
                    score += value
        return score

    df['score'] = df.apply(calculate_score, axis=1)
    df['risk'] = df['score'].map(mapping)
    return df

def predict_data(data, model_index):
    model_path = "model/model_{}.pkl".format(model_index)
    # Load object from the pickle file
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Drop account_id and email 
    if ('account_id' in data.columns or 'email' in data.columns):
        data = data.drop(['account_id','email'], axis=1)
    print(len(data.columns))
    X_test, y_test = preprocess_data(data, 'label')
    print(X_test.shape)
    print(X_test)
    print(y_test.shape)
    pred = get_prediction(model, X_test)
    model_dict = get_calculation_table(model)
    print("model_dict" , model_dict)
    predicted_label = pd.Series(pred.tolist())
    predicted_label.name = "label"
    mapped_predicted_label = predicted_label.map({1: "fraud", -1: "non-fraud"})
    columns = model_dict['features'] + ['score','risk']
    data = calculate_scores(data, model_dict)
    # Select only columns used in the risk score
    data = data[columns]
    # Apply mapping for the column values
    data = apply_mapping(data, model_dict["model"])

    full_data = pd.concat([data, mapped_predicted_label], axis=1)
    
    return full_data

def convert_string_format(input_string):
    # Split the string into individual words using underscores as separators
    words = input_string.split('_')

    # Iterate through the words and perform necessary formatting
    formatted_words = []
    for word in words:
        if word.isdigit():
            # If the word is a digit, add it as it is
            formatted_words.append(word)
        else:
            # Remove the "category" part and capitalize the first letter
            formatted_word = word.replace("category", "").strip().capitalize()
            formatted_words.append(formatted_word)

    # Join the formatted words with spaces
    formatted_string = ' '.join(formatted_words)
    formatted_string = formatted_string.strip()

    return formatted_string


def convert_df_to_json(df):
    prediction_list = df.to_dict(orient='records')
    # Convert JSON instances to Dataframe-like dictionaries
    dataframes = []
    count = 0
    for instance in prediction_list:
        count+=1
        feature_list = [x for x in instance.keys() if x not in ['account_id','email','score','risk','label']]
        # print("Keys",feature_list)
        feature_score = {}

        for feature in feature_list:
            # # Rename feature
            # feature_converted = convert_string_format(feature)

            alias = get_alias(feature)
            desc = get_description(feature)
            
            # feature_converted = feature

            # Create mapping
            feature_score[alias] = instance[feature]  # Set default value as None

            # TODO delete
            # feature_score[feature]["point"] = instance[feature] 
            # feature_score[feature]["alias"] = alias
            # feature_score[feature]["desc"] = desc

        # print("feature_score",feature_score)
        dataframe = {
            # "id": count,
            "account_id": instance["account_id"],
            "email": instance["email"],
            "feature_score": feature_score,
            "total_score": instance["score"],
            "risk_percentage": instance["risk"],
            "label":instance["label"],
        }
        dataframes.append(dataframe)
    return dataframes

def convert_prediction_list_to_dataframe(prediction_list):
    features = list(prediction_list[0]['feature_score'].keys())
    # features = [convert_string_format(x) for x in features]
        
    # headers = ["id", "total_score", "risk_percentage", "label"] + features
    headers = ["account_id", "email", "risk_percentage", "label"] + features + ["total_score",]

    # Collect all rows as a list of dictionaries
    rows = []
    for item in prediction_list:
        row = {
            "account_id": item['account_id'],
            "email": item['email'],
            "risk_percentage": item['risk_percentage'],
            "label": item['label'],
            "total_score": item['total_score']
        }
        feature_score = item['feature_score']
        row.update({key: feature_score.get(key, None) for key in features})
        rows.append(row)

    # Create the DataFrame in a single operation
    df = pd.DataFrame(rows, columns=headers)

    # # print('headers:',headers)
    # # Create an empty dataframe
    # df = pd.DataFrame(columns=headers)
    # # Iterate over the data and append rows to the dataframe
    # for item in prediction_list:
    #     # row = [item['id']]
    #     row = [item['account_id'],item['email'],item['risk_percentage'], item['label']]
    #     feature_score = item['feature_score']
    #     row += [feature_score.get(key, None) for key in features]
    #     row += [item['total_score']]
    #     df = df.append(pd.Series(row, index=headers), ignore_index=True)

    return df

def calculate_account_percentage(df):
    # Group by the 'label' column and count the occurrences of each label
    aggregated_df = df.groupby('label').size().reset_index(name='count')

    # Calculate the total count
    total_count = aggregated_df['count'].sum()

    # Calculate the percentage
    aggregated_df['percentage'] = round((aggregated_df['count'] / total_count) * 100,2)

    # Convert the percentage to string format with the percentage symbol
    aggregated_df['percentage'] = aggregated_df['percentage'].astype(str) + '%'

    # Rename the 'label' column to 'account_type'
    aggregated_df = aggregated_df.rename(columns={'label': 'label'})

    return aggregated_df

# TODO
def get_fraud_insight(df):
    # print(df.columns)
    df_fraud = df[df["label"] == "fraud"]

    # Get feature columns and label
    df_feature = df_fraud.iloc[:,-6:-1]

    agg_df = pd.DataFrame()
    feature_columns = df_feature.columns
    print("feature cols",feature_columns)
    for col in feature_columns:
        agg_df[col] = df_feature[col].apply(lambda x: 1 if x != 0 else 0)

    print("AGG DF")
    total_length = len(agg_df)

    percentages = pd.DataFrame()
    for col in agg_df.columns:
        if col != 'label':
            percentage = 0
            values_greater_than_1 = 0
            if total_length != 0:
                values_greater_than_1 = sum(1 for value in agg_df[col] if value > 0)
                percentage = (values_greater_than_1 / total_length) * 100
            print(col,":",percentage)

            percentages = percentages.append({'feature': col, 'countFraud':values_greater_than_1,'percentage': percentage}, ignore_index=True)

    sorted_percentages = percentages.sort_values(by='percentage', ascending=False)

    return sorted_percentages

# # TODO: train based on the best experiment
# train_fasterrisk_with_smote("data_e_smote_train")

train_fasterrisk_with_smote("final_data_train")

# # Load object from the pickle file
# with open("model/model.pkl", "rb") as f:
#     model = pickle.load(f)
#  # Get training data
# file_path = "dataset/" + "data_e" + "_test.csv"
# test_data = read_data(file_path)
# print(test_data.shape)
# print(test_data.head())
# df = predict_data(test_data)
# print(convert_df_to_json(df.head()))

### DUMP ###

# print(df.head().to_dict())
# json_list = df.head().to_dict(orient='records')
# print(json_list)
# # test_data = test_data.drop(['account_id'],axis=1)
# X_test, y_test = preprocess_data(test_data, 'label')
# pred = get_prediction(model, X_test)
# model_dict = get_calculation_table(model)
# # print(model_dict["features_list"])
# print(model_dict)
# predicted_label = pd.Series(pred.tolist())
# predicted_label.name = "label"
# mapped_predicted_label = predicted_label.map({1: "fraud", -1: "non-fraud"})
# print(mapped_predicted_label)
# data = calculate_scores(test_data, model_dict)
# full_data = pd.concat([data.drop(['label'],axis=1), mapped_predicted_label], axis=1)
# print(full_data.sample(25))
# print(full_data[(full_data['label'] == 'fraud') & (full_data['risk'] == '50.0%')])