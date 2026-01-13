import pandas as pd
from sklearn.model_selection import train_test_split

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
    a1 = df.drop(columns=df.columns[df.isnull().sum() > df.shape[0]*0.5])
    cols = [col for col in a1.columns if col not in ["Age", "Fare"]]
    l = []
    for col in cols:
        vals = a1[col].value_counts(dropna=False) > 0.7*a1.shape[0]
        if True in vals.values:
            l.append(col)
    a2 = a1.drop(columns=l)
    return a2.shape[1]

def Q3(df):
    '''
       Problem 3:
            Remove all rows with missing targets (the variable "Survived")
            How many rows do we have left?
    '''
    a1 = df.dropna(subset=["Survived"])
    return a1.shape[0]

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
    q1,_,q3 = df["Fare"].describe().values[4:7]
    iqr = q3-q1
    fare = df["Fare"].apply(lambda x: (q1-1.5*iqr) if x < q1-1.5*iqr else (q3+1.5*iqr) if x > q3+1.5*iqr else (x)  )
    average = fare.mean()
    return round(average,2)


def Q5(df):
    '''
       Problem 5:
            Impute missing value
            For number type column, impute missing values with mean
            What is the average (mean) of “Age” after imputing the missing values (round 2 decimal points)?
            Hint: Use function round(_, 2)
    '''
    avg = round(df["Age"].mean(),2)
    avg = df["Age"].fillna(avg).mean()
    return round(avg,2)


def Q6(df):
    '''
        Problem 6:
            Convert categorical to numeric values
            For the variable “Embarked”, perform the dummy coding.
            What is the average (mean) of “Embarked_Q” after performing dummy coding (round 2 decimal points)?
            Hint: Use function round(_, 2)
    '''
    for val in df["Embarked"].values:
        df["Embarked_"+str(val)] = df["Embarked"].apply(lambda x: 1 if x == val else 0)
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
    med = round(df["Survived"].mean(),2)
    df_copy = df.copy()
    df_copy = df_copy.fillna({"Survived":med})
    y = df_copy.pop("Survived")
    _, _, train_y, _ = train_test_split(df_copy,y,stratify=y, random_state=123, test_size=0.3)
    return round(train_y[train_y == 1].sum() / train_y.shape[0],2)
