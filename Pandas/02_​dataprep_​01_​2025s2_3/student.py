import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.pipeline import make_pipeline

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
    threshold = 0.5*df.shape[0]
    to_remove = df.isnull().sum()[df.isnull().sum() > threshold].index.tolist()
    cols = df.columns.tolist()
    for col in cols:
        if df[col].value_counts(normalize=True, dropna=False).iloc[0] >= 0.7:
            to_remove.append(col)
    df = df.drop(to_remove,axis=1)
    return df.shape[1]

def Q3(df):
    '''
       Problem 3:
            Remove all rows with missing targets (the variable "Survived")
            How many rows do we have left?
    '''
    return df.loc[~df['Survived'].isnull(),:].shape[0]

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
    q1, _, q3 = df["Fare"].describe().iloc[4:7].values
    iqr = q3-q1
    left, right = [q1 - 1.5*iqr, q3 + 1.5*iqr]
    df["Fare"] = df["Fare"].apply(lambda x: left if x < left else (right if x > right else x))
    return round(df["Fare"].mean(), 2)

def Q5(df):
    '''
       Problem 5:
            Impute missing value
            For number type column, impute missing values with mean
            What is the average (mean) of “Age” after imputing the missing values (round 2 decimal points)?
            Hint: Use function round(_, 2)
    '''
    df = df.fillna(df.mean(numeric_only=True))    
    return round(df["Age"].mean(),2)

def Q6(df):
    '''
        Problem 6:
            Convert categorical to numeric values
            For the variable “Embarked”, perform the dummy coding.
            What is the average (mean) of “Embarked_Q” after performing dummy coding (round 2 decimal points)?
            Hint: Use function round(_, 2)
    '''
    df = pd.get_dummies(df, df.select_dtypes(include='object').columns.tolist())
    return round(df["Embarked_Q"].mean(),2)

def Q7(df):
    '''
        Problem 7:
            Split train/test split with stratification using 70%:30% and random seed with 123
            Show a proportion between survived (1) and died (0) in all data sets (total data, train, test)
            What is the proportion of survivors (survived = 1) in the training data (round 2 decimal points)?
            Hint: Use function round(_, 2), and train_test_split() from sklearn.model_selection, 
            Don't forget to impute missing values with mean.
    '''
    df = df.loc[~df["Survived"].isnull(), :]
    y = df.pop("Survived")
    x = df.copy()
    train_x, test_x, train_y, test_y = train_test_split(x,y, stratify=y, test_size=0.3, random_state=123)
    def handle_outlieir(X):
        X = X.copy()
        q1, _, q3 = X["Fare"].describe().iloc[4:7].values
        iqr = q3-q1
        left, right = [q1 - 1.5*iqr, q3 + 1.5*iqr]
        X["Fare"] = X["Fare"].apply(lambda x: left if x < left else (right if x > right else x))
        return X
    OutlierTransform = FunctionTransformer(handle_outlieir)
    preprocesses = make_column_transformer(
        ('drop',['Cabin', 'Parch','Name', 'Ticket']),
        (OutlierTransform, ["Fare"]),
        (SimpleImputer(strategy='mean'), make_column_selector(dtype_include="number")),
        (OneHotEncoder(sparse_output=False), make_column_selector(dtype_include="object")),
        remainder='passthrough'
    )
    pipeline = make_pipeline(preprocesses)
    pipeline.fit(train_x, train_y)
    return round(train_y.loc[train_y == 1].count()/train_y.shape[0],2)

    
