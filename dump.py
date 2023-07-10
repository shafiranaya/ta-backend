
# def fasterrisk_with_smote(data, sparsity=5, parent_size=10,model_index=0, random_state=0):
#     df = data.copy()
#     target_col = 'label'
#     df = preprocess_data(df,target_col)

#     # Splitting the data
#     X = df.drop(['label'], axis=1)
#     y = df['label']

#     # X_train = pd.read_csv("../dataset/data_b_train.csv")
#     # X_test = pd.read_csv("../dataset/data_b_test")
#     # X_train, y_train = get_dataset("data_f_smote")
#     train_data = read_data("dataset/data_f_smote_train.csv")
#     train_data = preprocess_data(train_data,'label')
#     print(train_data['label'].value_counts())
#     print(train_data['label'].unique())
#     train_data = np.asarray(train_data)
#     X_train, y_train = train_data[:, :-1], train_data[:, -1]
#     # X_train, y_train, X_val, y_val, X_test, y_test = get_dataset("data_f_smote")
#     # # Split the data into training, validation, and test sets
#     # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
#     # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=random_state)

#     # # Separate minority and majority classes
#     # minority_class = df[df['label'] == 1]
#     # majority_class = df[df['label'] == -1]
    
#     # print(len(majority_class), len(minority_class))
#     # print(len(X_train), len(X_val), len(X_test))
#     # print(len(y_train), len(y_val), len(y_test))
#     # Resample
#     # start_sampling_time = time.time()
#     # # smote = SMOTE()
#     # # X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
#     # stop_sampling_time = time.time()
    
#     ### CONVERT TO NUMPY ###
#     # X_train = np.asarray(X_train)
#     # y_train = np.asarray(y_train)
#     # X_val = np.asarray(X_val)
#     # y_val = np.asarray(y_val)
#     # X_test = np.asarray(X_test)
#     # y_test = np.asarray(y_test)
#     # X_test_imbalanced = np.asarray(X_test_imbalanced)
#     # y_test_imbalanced = np.asarray(y_test_imbalanced)

#     ### MODELLING ###
#     RiskScoreOptimizer_m = RiskScoreOptimizer(X = X_train, y = y_train, k = sparsity, parent_size = parent_size)
#     # start_training_time = time.time()
#     RiskScoreOptimizer_m.optimize()
#     # stop_training_time = time.time()
#     # training_time = stop_training_time - start_training_time

#     multipliers, sparseDiversePool_beta0_integer, sparseDiversePool_betas_integer = RiskScoreOptimizer_m.get_models()
#     # print("We generate {} risk score models from the sparse diverse pool".format(len(multipliers)))
    
#     model_index = 0 # first model
#     multiplier = multipliers[model_index]
#     intercept = sparseDiversePool_beta0_integer[model_index]
#     coefficients = sparseDiversePool_betas_integer[model_index]
#     model = RiskScoreClassifier(multiplier, intercept, coefficients)
#     X_featureNames = list(X.columns)

#     model.reset_featureNames(X_featureNames)
#     model.print_model_card()
#     print("Model",model)
#     # Dump object into a pickle file
#     with open("model.pkl", "wb") as f:
#         pickle.dump(model, f)
#     # # Load object from the pickle file
#     # with open("model.pkl", "rb") as f:
#     #     loaded_data = pickle.load(f)
    
#     return model
#     # sampling_time = stop_sampling_time - start_sampling_time
#     # # Print the number of examples in each set
#     # print("Number of examples in the training set: ", len(X_train))
#     # print("Number of examples in the validation set: ", len(X_val))
#     # print("Number of examples in the test set: ", len(X_test))

# def get_dataset(dataset_name):
#     # dataset_name ='data_f'
#     train_data_file_path = "dataset/"+ dataset_name + "_train.csv"
#     # test_data_file_path = "dataset/"+ dataset_name + "_test.csv"
#     # val_data_file_path = "dataset/"+ dataset_name + "_val.csv"
#     # test_imbalanced_data_file_path = "../dataset/"+ dataset_name + "_test_imbalanced.csv"
#     train_data = pd.read_csv(train_data_file_path)
#     # test_data = pd.read_csv(test_data_file_path)
#     # val_data = pd.read_csv(val_data_file_path)

#     X_train = train_data.drop(['label'],axis=1)
#     y_train = train_data['label']
#     # X_test = test_data.drop(['label','account_id'],axis=1)
#     # y_test = test_data['label']
#     # X_val = val_data.drop(['label','account_id'],axis=1)
#     # y_val = val_data['label']
#     return X_train, y_train
#     # , X_val, y_val, X_test, y_test