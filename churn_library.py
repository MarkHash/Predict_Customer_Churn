# library doc string
"""
Library for churn_notebook
Author: Mark

Date: Nov 2022
"""

# import libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.model_selection import train_test_split
os.environ['QT_QPA_PLATFORM']='offscreen'



def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    return pd.read_csv(pth)


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    
    # histgram of churn
    df['Churn'].hist()
    plt.savefig('./images/eda/churn_hist.png')
    plt.close('churn_hist.png')

    # histogram of Customer_Age
    df['Customer_Age'].hist()
    plt.savefig('./images/eda/Customer_Age.png')
    plt.close('Customer_Age.png')

    # histogram of Marital_Status
    df['Marital_Status'].value_counts('normalize').plot(kind='bar')
    plt.savefig('./images/eda/Marital_Status.png')
    plt.close('Marital_Status.png')

    # histogram of Total_Trans_Ct
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig('./images/eda/Total_Trans_Ct.png')
    plt.close('Total_Trans_Ct.png')

    # heatmap
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
    plt.savefig('./images/eda/heatmap.png')
    plt.close('heatmap.png')

def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    for category_name in category_lst:
        lst = []
        column_name = category_name + response
        category_groups = df.groupby(category_name).mean()['Churn']
        for val in df[category_name]:
            lst.append(category_groups.loc[val])

        df[column_name] = lst
    return df



def perform_feature_engineering(df, cols, response):
    '''
    input:
              df: pandas dataframe
              cols: list of columns that will be kept for model training
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    X = pd.DataFrame()
    X[cols] = df[cols]
    y = df[response]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    pass


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    pass

def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    pass

if __name__ == "__main__":
        cat_columns = [
                'Gender',
                'Education_Level',
                'Marital_Status',
                'Income_Category',
                'Card_Category'
                ]
        column_post_fix = '_Churn'
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
        df = import_data("./data/bank_data.csv")
        perform_eda(df)
        df = encoder_helper(df, cat_columns, column_post_fix)
        X_train, X_test, y_train, y_test = perform_feature_engineering(df, keep_cols, 'Churn')
