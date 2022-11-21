import os
import logging
import churn_library_solution as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def test_import(import_data):
	'''
	test data import - this example is completed for you to assist with the other test functions
	'''
	try:
		df = import_data("./data/bank_data.csv")
		logging.info("Testing import_data: SUCCESS")
	except FileNotFoundError as err:
		logging.error("Testing import_eda: The file wasn't found")
		raise err

	try:
		assert df.shape[0] > 0
		assert df.shape[1] > 0
	except AssertionError as err:
		logging.error("Testing import_data: The file doesn't appear to have rows and columns")
		raise err


def test_eda(perform_eda):
	'''
	test perform eda function
	'''
	try:
		assert os.path.exists('./images/eda/churn_hist.png')
		assert os.path.exists('./images/eda/Customer_Age.png')
		assert os.path.exists('./images/eda/Marital_Status.png')
		assert os.path.exists('./images/eda/Total_Trans_Ct.png')
		assert os.path.exists('./images/eda/heatmap.png')
		logging.info("Testing perform_eda: SUCCESS")
	except AssertionError as err:
		logging.error("Testing perform_eda: image files don't appear to be generated")
		raise err



def test_encoder_helper(encoder_helper):
	'''
	test encoder helper
	'''


def test_perform_feature_engineering(perform_feature_engineering):
	'''
	test perform_feature_engineering
	'''


def test_train_models(train_models):
	'''
	test train_models
	'''


if __name__ == "__main__":
	pass








