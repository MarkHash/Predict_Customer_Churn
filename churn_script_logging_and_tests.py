import os
import logging
import churn_library as clib
# import churn_library_solution as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def test_import():
	'''
	test data import - function to test perform_eda() function
	'''
	try:
		df = clib.import_data("./data/bank_data.csv")
		logging.info("Testing import_data: SUCCESS")
	except FileNotFoundError as err:
		logging.error("Testing import_eda: The file wasn't found")
		raise err

	try:
		assert df.shape[0] > 0
		assert df.shape[1] > 0
		return df
	except AssertionError as err:
		logging.error("Testing import_data: The file doesn't appear to have rows and columns")
		raise err


def test_eda(df):
	'''
	test perform eda function - function to test import_data() function
	'''
	try:
		clib.perform_eda(df)
		assert os.path.exists('./images/eda/churn_hist.png')
		assert os.path.exists('./images/eda/Customer_Age.png')
		assert os.path.exists('./images/eda/Marital_Status.png')
		assert os.path.exists('./images/eda/Total_Trans_Ct.png')
		assert os.path.exists('./images/eda/heatmap.png')
		logging.info("Testing perform_eda: SUCCESS")
	except AssertionError as err:
		logging.error("Testing perform_eda: image files don't appear to be generated")
		raise err



def test_encoder_helper(df, category_lst, response):
	'''
	test encoder helper - function to test encoder_helper() function
	'''
	try:
		encoded_df = clib.encoder_helper(df, category_lst, response)
		original_df = clib.import_data("./data/bank_data.csv")
		assert (len(encoded_df.columns) == (len(original_df.columns) + len(category_lst) + 1))
		logging.info("Testing encoder_helper: SUCCESS")
	except AssertionError as err:
		logging.error("Testing encoder_helper: The dataframe doesn't appear to have additional encoded columns")
		raise err

def test_perform_feature_engineering(df, cols, response):
	'''
	test perform_feature_engineering - function to test perform_feature_engineering() function
	'''
	try:
		X_train, X_test, y_train, y_test = clib.perform_feature_engineering(df, cols, response)
		assert X_train.shape[0] > 0
		assert X_test.shape[0] > 0
		assert y_train.shape[0] > 0
		assert y_test.shape[0] > 0
		logging.info("Testing perform_feature_engineering: SUCCESS")
		return X_train, X_test, y_train, y_test
	except AssertionError as err:
		logging.error("Testing perform_feature_engineering: The dataframes appear to be empty")
		raise err


def test_train_models(X_train, X_test, y_train, y_test):
	'''
	test train_models - function to test train_models() function
	'''
	try:
		clib.train_models(X_train, X_test, y_train, y_test)
		assert os.path.exists('./images/results/roc_curve.png')
		assert os.path.exists('./images/results/feature_importance.png')
		assert os.path.exists('./images/results/Random_Forest.png')
		assert os.path.exists('./images/results/Logistic_Regression.png')
		assert os.path.exists('./models/rfc_model.pkl')
		assert os.path.exists('./models/logistic_model.pkl')
		logging.info("Testing train_models: SUCCESS")
	except AssertionError as err:
		logging.error("Testing train_models: model and/or image files don't appear to be generated")
		raise err

if __name__ == "__main__":
	cat_columns = [
		'Gender',
		'Education_Level',
		'Marital_Status',
		'Income_Category',
		'Card_Category'
                ]
	keep_cols = [
		'Customer_Age',
		'Dependent_count',
		'Months_on_book',
		'Total_Relationship_Count', 'Months_Inactive_12_mon',
		'Contacts_Count_12_mon',
		'Credit_Limit',
		'Total_Revolving_Bal',
		'Avg_Open_To_Buy',
		'Total_Amt_Chng_Q4_Q1',
		'Total_Trans_Amt',
		'Total_Trans_Ct',
		'Total_Ct_Chng_Q4_Q1',
		'Avg_Utilization_Ratio',
		'Gender_Churn',
		'Education_Level_Churn',
		'Marital_Status_Churn',
		'Income_Category_Churn',
		'Card_Category_Churn'
		]
	df = test_import()
	test_eda(df)
	test_encoder_helper(df, cat_columns, '_Churn')
	X_train, X_test, y_train, y_test = test_perform_feature_engineering(df, keep_cols, 'Churn')
	test_train_models(X_train, X_test, y_train, y_test)
