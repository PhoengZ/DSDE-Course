import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
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
    for col in df.columns.tolist():
        if df[col].isnull().sum() > 0.5*df.shape[0]:
            df = df.drop(col, axis=1)
    for col in df.columns.tolist():
        if col in ["Age", "Fare"]:continue
        if df[col].value_counts(normalize=True, dropna=False).iloc[0] > 0.7:
            df = df.drop(col, axis=1)
    return df.shape[1]

def Q3(df):
    '''
       Problem 3:
            Remove all rows with missing targets (the variable "Survived")
            How many rows do we have left?
    '''
    df = df.loc[~df['Survived'].isnull(), :]
    return df.shape[0]

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
    q1, _, q3 = df["Fare"].describe().values[4:7]
    iqr = q3 - q1
    lower_bound, upper_bound = [q1 - 1.5*iqr, q3 + 1.5*iqr]
    df["Fare"] = df["Fare"].apply(lambda x: lower_bound if x < lower_bound else (upper_bound if x > upper_bound else x))
    return round(df["Fare"].mean(),2)

def Q5(df):
    '''
       Problem 5:
            Impute missing value
            For number type column, impute missing values with mean
            What is the average (mean) of “Age” after imputing the missing values (round 2 decimal points)?
            Hint: Use function round(_, 2)
    '''
    nsimp = SimpleImputer(strategy="mean")
    ncols = df.select_dtypes(include='number').columns
    df[ncols] = nsimp.fit_transform(df[ncols])
    return round(df["Age"].mean(), 2)

def Q5_1(df):
    df = df.fillna(df.mean(numeric_only=True))
    return round(df["Age"].mean(), 2)

def Q6(df):
    '''
        Problem 6:
            Convert categorical to numeric values
            For the variable “Embarked”, perform the dummy coding.
            What is the average (mean) of “Embarked_Q” after performing dummy coding (round 2 decimal points)?
            Hint: Use function round(_, 2)
    '''
    onehot = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cols = df.select_dtypes(include="object").columns
    new_df = pd.DataFrame(onehot.fit_transform(df[cols]), index=df.index, columns=onehot.get_feature_names_out())
    df = df.drop(cols, axis=1)
    df = pd.concat([df, new_df], join="inner", axis=1)
    return round(df["Embarked_Q"].mean(),2)

def Q6_1(df):
    df = pd.get_dummies(df, df.select_dtypes(include='object').columns)
    # Shape ต่างกับ Q6 เพราะ get_dummies ไม่สร้าง columnns onehot กับ value NaN แต่ oneHot สร้าง
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
    y = df.pop("Survived")
    x = df.copy()
    _, _, train_y, _ = train_test_split(x, y, stratify=y, random_state=123, test_size=0.3)
    return round(train_y[train_y == 1].shape[0]/train_y.shape[0],2)
