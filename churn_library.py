"""
Author: Bibin MJ
Script for EDA, preprocess data, generate churn model
"""

# import libraries
from typing import List
import seaborn as sns
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import shap
import logging
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
logging.basicConfig(
    filename='.logs/info.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

sns.set()


cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category',
    ]
quant_columns = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio'
    ]


def import_data(pth: str) -> pd.DataFrame:
    """Returns dataframe for the csv found at pth

    Args:
            pth (str): path to csv file

    Returns:
            pd.DataFrame: pandas dataframe from csv
    """
    return pd.read_csv(pth)


def perform_eda(df: pd.DateFrame) -> None:
    """perform eda on df and save figures to images folder

    Args:
            df (pd.DateFrame): input data frame
    output:
            None
    """
    logging.info(f'Shape of data frame: {df.shape}')
    logging.info(f'Nulls in columns: {df.isnull().sum()}')
    logging.info(f'Summary Stats: {df.describe()}')

    # define Churn column
    df['Churn'] = df['Attrition_Flag`'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    # make and save EDA plots
    # churn histogram
    fig, ax = plt.figure(figsize=(20, 10))
    df['Churn'].hist(ax=ax)
    img_pth = 'images/eda/churn_histogram.png'
    fig.savefig(img_pth)
    logging.info(f'Saving file to {img_pth}')
    plt.close(fig)

    # customer age
    fig, ax = plt.figure(figsize=(20, 10))
    df['Customer_Age'].hist(ax=ax)
    img_pth = 'images/eda/customer_age.png'
    fig.savefig(img_pth)
    logging.info(f'Saving file to {img_pth}')
    plt.close(fig)

    # marital status
    fig, ax = plt.figure(figsize=(20, 10))
    df['Marital_Status'].value_counts('normalize').plot(kind='bar', ax=ax)
    img_pth = 'images/eda/churn_histogram.png'
    fig.savefig(img_pth)
    logging.info(f'Saving file to {img_pth}')
    plt.close(fig)

    # total transactions
    fig, ax = plt.figure(figsize=(20, 10))
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True, ax=ax)
    img_pth = 'images/eda/churn_histogram.png'
    fig.savefig(img_pth)
    logging.info(f'Saving file to {img_pth}')
    plt.close(fig)

    # correlation
    fig, ax = plt.figure(figsize=(20, 10))
    sns.heatmap(
        df.corr(),
        annot=False,
        cmap='Dark2_r',
        linewidths=2,
        ax=ax)
    fig.savefig(img_pth)
    logging.info(f'Saving file to {img_pth}')
    plt.close(fig)


def encoder_helper(
    df: pd.DataFrame,
    category_lst: List,
     response: str) -> pd.DataFrame:
	"""Helper function to turn each categorical column into a new column with
    propotion of churn for each category
	- associated with cell 15 from the notebook

	Args:
		df (pd.DataFrame): input dataframe
		category_lst (List): list of columns that contain categorical features
		response (str): string of response name [optional argument
		that could be used for naming variables or index y column]
	Returns:
		pd.DataFrame: pandas dataframe with new columns for
	"""
    pass


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''


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
