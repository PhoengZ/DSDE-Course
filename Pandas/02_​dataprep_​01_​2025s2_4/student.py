import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
"""
    ASSIGNMENT 2 (STUDENT VERSION):
    Using pandas to explore Titanic data from Kaggle (titanic_to_student.csv) and answer the questions.
    (Note that the following functions already take the Titanic dataset as a DataFrame, so you don’t need to use read_csv.)

"""


def Q1(df):
    """
        Problem 1:
            How many rows are there in the "titanic_to_student.csv"?
    """
    return df.shape[0]

def Q2(df):
    '''
        Problem 2:
            2.1 Drop variables with missing > 50%
            2.2 Check all columns except 'Age' and 'Fare' for flat values, drop the columns where flat value > 70%
            From 2.1 and 2.2, how many columns do we have left?
            Note: 
            -Ensure missing values are considered in your calculation. If you use normalize in .value_counts(), please include dropna=False.
    '''
    class DropWMissing(BaseEstimator, TransformerMixin):
        def __init__(self, threshold=0.5):
            self.threshold = threshold
        def fit(self, X, y=None):
            X = X.copy()
            self.featurein_ = X.columns.tolist()
            vals = X.isnull().sum()
            vals = vals[vals > self.threshold*X.shape[0]]
            self.remove_cols_ = vals.index.tolist()
            return self
        def transform(self, X):
            X = X.copy()
            X = X.drop(self.remove_cols_, axis=1)
            return X
        def get_feature_names_out(self):
            return [col for col in self.featurein_ if col not in self.remove_cols_]
    class DropWflatValues(BaseEstimator, TransformerMixin):
        def __init__(self, threshold=0.7):
            self.threshold = threshold
        def fit(self, X, y=None):
            X = X.copy()
            self.featurein = X.columns.tolist()
            self.remove_cols_ = []
            for col in X.columns.tolist():
                if col in ["Age", "Fare"]:
                    continue
                if X[col].value_counts(normalize=True, dropna=False).iloc[0] > self.threshold:
                    self.remove_cols_.append(col)
            return self
        def transform(self, X):
            X = X.copy()
            X = X.drop(self.remove_cols_, axis=1)
            return X
        def get_feature_names_out(self):
            return [col for col in self.featurein_ if col not in self.remove_cols_]
    pipeline = make_pipeline(
        (DropWMissing(threshold=0.5)),
        (DropWflatValues(threshold=0.7)),
    )
    pipeline.set_output(transform="pandas")
    pipeline.fit(df)
    new_df = pipeline.transform(df)
    return new_df.shape[1]

def Q3(df):
    '''
       Problem 3:
            Remove all rows with missing targets (the variable "Survived")
            How many rows do we have left?
    '''
    class DropTarget(BaseEstimator, TransformerMixin):
        def __init__(self, target_name="Survived"):
            self.target = target_name 
        def fit(self, X, y=None):
            X = X.copy()
            self.feature_in_ = X.columns.tolist()
            self.keep_cols_ = X.loc[~X[self.target].isnull(), :].index.tolist() 
            return self
        def transform(self, X):
            X = X.copy()
            X = X.iloc[self.keep_cols_, :]
            return X
        def get_feature_names_out(self):
            return self.feature_in_
    pipeline = make_pipeline(
        (DropTarget(target_name="Survived"))
    )
    pipeline.set_output(transform="pandas")
    pipeline.fit(df)
    new_df = pipeline.transform(df)
    return new_df.shape[0]

def Q4(df):
    '''
       Problem 4:
            Handle outliers
            For the variable “Fare”, replace outlier values with the boundary values
            If value < (Q1 - 1.5IQR), replace with (Q1 - 1.5IQR)
            If value > (Q3 + 1.5IQR), replace with (Q3 + 1.5IQR)
            What is the mean of “Fare” after replacing the outliers (round 2 decimal points)?
            Hint: Use function round(_, 2)
    '''
    class OutlierHandle(BaseEstimator, TransformerMixin):
        def __init__(self, features):
            self.features = features
        def fit(self, X, y=None):
            X = X.copy()
            self.featurein = X.columns.tolist()
            self.q1_, _, self.q3_ = X[self.features].describe().iloc[4:7].values
            self.iqr_ = self.q3_ - self.q1_
            return self
        def transform(self, X):
            X = X.copy()
            for index, col in enumerate(self.features):
                v1 = self.q1_ - 1.5*self.iqr_
                v2 = self.q3_ + 1.5*self.iqr_
                X[col] = X[col].apply(lambda x: v1[index] if x < v1[index] else (v2[index] if x > v2[index] else x))
            return X
        def get_feature_names_out(self):
            return self.featurein
    pipeline = make_pipeline(
        (OutlierHandle(features=["Fare"]))
    )
    pipeline.set_output(transform="pandas")
    pipeline.fit(df)
    new_df =  pipeline.transform(df)
    return round(new_df["Fare"].mean(),2)

def Q5(df):
    '''
       Problem 5:
            Impute missing value
            For number type column, impute missing values with mean
            What is the average (mean) of “Age” after imputing the missing values (round 2 decimal points)?
            Hint: Use function round(_, 2)
    '''
    cate_cols = df.select_dtypes(include='object').columns.tolist()
    num_cols = df.select_dtypes(include='number').columns.tolist()
    cate_pipe = make_pipeline(
        (SimpleImputer(strategy="most_frequent"))
    )   
    num_pipe = make_pipeline(
        (SimpleImputer(strategy="mean"))
    )
    pipeline = ColumnTransformer([
        ("cate_pipe",cate_pipe, cate_cols),
        ("num_pipe",num_pipe, num_cols),
    ], remainder="passthrough")
    pipeline.set_output(transform="pandas")
    pipeline.fit(df)
    new_df = pipeline.transform(df)
    return round(new_df["num_pipe__Age"].mean(), 2)

def Q6(df):
    '''
        Problem 6:
            Convert categorical to numeric values
            For the variable “Embarked”, perform the dummy coding.
            What is the average (mean) of “Embarked_Q” after performing dummy coding (round 2 decimal points)?
            Hint: Use function round(_, 2)
    '''
    cate_pipe = make_pipeline(
        (OneHotEncoder(sparse_output=False))
    )
    cate_cols = df.select_dtypes(include='object').columns.tolist()
    pipeline = ColumnTransformer([
        ("cate_pipe",cate_pipe, cate_cols)
    ], remainder="passthrough")
    pipeline.set_output(transform="pandas")
    pipeline.fit(df)
    new_df = pipeline.transform(df)
    return round(new_df["cate_pipe__Embarked_Q"].mean(),2)

def Q7(df):
    '''
        Problem 7:
            Split train/test split with stratification using 70%:30% and random seed with 123
            Show a proportion between survived (1) and died (0) in all data sets (total data, train, test)
            What is the proportion of survivors (survived = 1) in the training data (round 2 decimal points)?
            Hint: Use function round(_, 2), and train_test_split() from sklearn.model_selection, 
            Don't forget to impute missing values with mean.
    '''
    y = df.pop("Survived")
    x = df.copy()
    train_x, test_x, train_y, test_y = train_test_split(x, y, stratify=y, random_state=123, test_size=0.3)
    return round(train_y[train_y == 1].sum()/ train_y.count(),2)