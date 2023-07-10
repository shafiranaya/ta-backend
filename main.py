from fastapi import FastAPI, Response, params
from fastapi.params import Query
from fastapi.middleware.cors import CORSMiddleware
import csv
from func import calculate_account_percentage, convert_df_to_json, convert_prediction_list_to_dataframe, get_fraud_insight, predict_data, preprocess_data, train_fasterrisk_with_smote, get_calculation_table, read_data, sort_data, search_data
import json
import io
import pandas as pd
# import joblib
import pickle

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["x-total-count"]
)


def perform_prediction(model_index: int):
    # TODO change
    file_path_test_data = "dataset/" + "final_data" + "_test.csv"
    test_data = read_data(file_path_test_data)
    identity_data = test_data[['account_id','email']]
    prediction_df = predict_data(test_data, model_index)
    prediction_df = pd.concat([prediction_df, identity_data], axis=1)
    prediction_list = convert_df_to_json(prediction_df)
    return prediction_list


@app.get("/")
async def root():
    return {"message": "Hello World"}

# Function to load the model
@app.get("/model/{model_index}")
async def get_model(model_index: int):
    # Validate the model index
    if model_index < 1 or model_index > 5:
        return {"error": "Invalid model index"}

    # Load object from the pickle file based on the model index
    model_file = f"model/model_{model_index}.pkl"
    with open(model_file, "rb") as f:
        model = pickle.load(f)

    table = get_calculation_table(model)
    return table

# Function to get distribution of features indicating fraud
@app.get("/insight/{model_index}")
async def get_insight(model_index: int):
    # Validate the model index
    if model_index < 1 or model_index > 5:
        return {"error": "Invalid model index"}

    prediction_list = perform_prediction(model_index)
    df = convert_prediction_list_to_dataframe(prediction_list)
    result_dict = get_fraud_insight(df).to_dict(orient="records")
    return result_dict

# Function to get summary count of the predicted label
@app.get("/summary/{model_index}")
async def get_summary(model_index: int):
    # Validate the model index
    if model_index < 1 or model_index > 5:
        return {"error": "Invalid model index"}

    file_path = "dataset/" + "data_e" + "_test.csv"
    test_data = read_data(file_path)
    # X_test, y_test = preprocess_data(test_data, 'label')
    # print("X_test",X_test.head())
    # Perform prediction on the paginated data
    prediction_list = perform_prediction(model_index)
    df = convert_prediction_list_to_dataframe(prediction_list)
    aggregate = calculate_account_percentage(df)
    json_data_agg = aggregate.to_dict(orient='records')
    result_dict = {}
    for item in json_data_agg:
        label = item["label"]
        count = item["count"]
        percentage = item["percentage"]
        result_dict[label] = {"metrics": item["label"], "count": count, "percentage": percentage}

    return result_dict

# Function to get prediction
@app.get("/prediction")
async def get_prediction(
    model_index: int,
    _page: int = Query(1, gt=0),
    _limit: int = Query(20, gt=0),
    _sort: str = Query("risk_percentage"),
    _order: str = Query("desc", regex="^(asc|desc)$"),
    _search: str = Query(None)  # New query parameter for search
):
    
    # Validate the model index
    if model_index < 1 or model_index > 5:
        return {"error": "Invalid model index"}

    # file_path = "dataset/" + "data_e" + "_test.csv"
    # test_data = read_data(file_path)
    # X_test, y_test = preprocess_data(test_data, 'label')
    # print("X_test",X_test.head())
    # Perform prediction on the paginated data
    # prediction_df = predict_data(test_data, model_index)
    # print(prediction_df.head())
    # prediction_list = convert_df_to_json(prediction_df)
    # print(prediction_list[:5])


    prediction_list = perform_prediction(model_index)
    print(prediction_list[:5])
     # Apply search filter if _search parameter is provided
    if _search:
        prediction_list = search_data(prediction_list, _search)  # Implement your search logic here
    print(prediction_list[:5])

    # Apply pagination and sorting
    start_index = (_page - 1) * _limit
    end_index = start_index + _limit
    sorted_data = sort_data(prediction_list, _sort, _order)
    paginated_data = sorted_data[start_index:end_index]

    total_count = len(prediction_list)
    print("TOTAL DATA: ", total_count)

    # prediction_df = predict_data(test_data)
    # prediction_list = convert_df_to_json(prediction_df)
    # print(prediction_list)
    # total_count = len(prediction_list)
    response = Response(
                content=json.dumps(paginated_data),
                headers={"x-total-count": str(total_count),
                },
                media_type="application/json"
    )
    return response
    # print("PREDICTION LIST",prediction_list)
    # return {"prediction_list":prediction_list}

@app.get("/export-csv")
async def export_csv(
    # _page: int = Query(1, gt=0),
    # _limit: int = Query(20, gt=0),
    model_index: int,
    _sort: str = Query("risk_percentage"),
    _order: str = Query("desc", regex="^(asc|desc)$"),
    _search: str = Query(None)  # New query parameter for search
):
    # file_path = "dataset/" + "data_e" + "_test.csv"
    # test_data = read_data(file_path)
    # # Perform prediction on the entire dataset
    # prediction_df = predict_data(test_data, model_index)
    # prediction_list = convert_df_to_json(prediction_df)

    prediction_list = perform_prediction(model_index)
    # Apply search filter if _search parameter is provided
    if _search:
        prediction_list = search_data(prediction_list, _search)
        prediction_list = sort_data(prediction_list,_sort,_order)

    print(prediction_list)
    
    df = convert_prediction_list_to_dataframe(prediction_list)
    # Create an in-memory file object to write the CSV data
    csv_file = io.StringIO()

    # Write data to the CSV file
    df.to_csv(csv_file, index=False)

    # Get the CSV file contents as a string
    csv_data = csv_file.getvalue()
    # # Create an in-memory file object to write the CSV data
    # csv_file = io.StringIO()
    # csv_writer = csv.writer(csv_file)

    # # Write headers to the CSV file
    # features = prediction_list[0]['feature_score'].keys()
    # headers = ["id", "risk_percentage", "label"] + features
    # csv_writer.writerow(headers)

    # # Write data rows to the CSV file
    # for row in prediction_list:
    #     csv_writer.writerow([row["id"], row["risk_percentage"], row["label"]])

    # Get the CSV data as a string
    csv_data = csv_file.getvalue()

    # Create a response with the CSV data
    response = Response(content=csv_data, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=export.csv"

    return response

# Store the blocked account IDs
blocked_accounts = set()

@app.put("/block/{account_id}")
def block_account(account_id: int):
    if account_id in blocked_accounts:
        return {"status": "Blocked"}
    
    blocked_accounts.add(account_id)
    print("Blocked accounts", blocked_accounts)

    return {"status": "Blocked"}

@app.put("/unblock/{account_id}")
def unblock_account(account_id: int):
    if account_id not in blocked_accounts:
        return {"status": "Allowed"}
    
    blocked_accounts.remove(account_id)
    print("Blocked accounts", blocked_accounts)

    return {"status": "Allowed"}

@app.get("/status/{account_id}")
def check_status(account_id: int):
    if account_id in blocked_accounts:
        return {"status": "Blocked"}
    
    return {"status": "Allowed"}


# @app.get("/model")
# async def get_model():
#     data = read_data("dataset/data_b.csv")
#     model = fasterrisk_with_smote(data)
#     table = get_calculation_table(model)
#     multiplier = model.multiplier
#     intercept = model.intercept
#     coefficients = model.coefficients
#     featureNames = model.featureNames
#     return table
    # multiplier, intercept, coefficients, featureNames = None
    # return {"model": {
    #     "multiplier": multiplier,
    #     "intercept":intercept,
    #     "coefficients":coefficients,
    #     "featureNames":featureNames,
    # }}

# @app.get("/data")
# async def get_data():
#     return 