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
		print("{} {} {}".format(len(encoded_df.columns), len(original_df.columns), len(category_lst)))
		assert (len(encoded_df.columns) == (len(original_df.columns) + len(category_lst) + 1))
		logging.info("Testing encoder_helper: SUCCESS")
	except AssertionError as err:
		logging.error("Testing encoder_helper: The dataframe doesn't appear to have additional encoded columns")

def test_perform_feature_engineering(perform_feature_engineering):
	'''
	test perform_feature_engineering - function to test perform_feature_engineering() function
	'''


def test_train_models(train_models):
	'''
	test train_models - function to test train_models() function
	'''


if __name__ == "__main__":
	cat_columns = [
                'Gender',
                'Education_Level',
                'Marital_Status',
                'Income_Category',
                'Card_Category'
                ]
	column_post_fix = '_Churn'
	df = test_import()
	test_eda(df)
	test_encoder_helper(df, cat_columns, column_post_fix)
