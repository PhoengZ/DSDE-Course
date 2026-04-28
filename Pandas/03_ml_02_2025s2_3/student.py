
import warnings # DO NOT modify this line
import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning # DO NOT modify this line
warnings.filterwarnings("ignore", category=ConvergenceWarning) # DO NOT modify this line


class BankLogistic:
    def __init__(self, data_path): # DO NOT modify this line
        self.data_path = data_path
        self.df = pd.read_csv(data_path, sep=',')
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def Q1(self): # DO NOT modify this line
        """
        Problem 1:
            Load ‘bank-st.csv’ data from the “Attachment”
            How many rows of data are there in total?

        """
        return self.df.shape[0]

    def Q2(self): # DO NOT modify this line
        """
        Problem 2:
            return the tuple of numeric variables and categorical variables are presented in the dataset.
        """
        return (len(self.df.select_dtypes(include="number").columns), len(self.df.select_dtypes(include='object').columns))
    
    def Q3(self): # DO NOT modify this line
        """
        Problem 3:
            return the tuple of the Class 0 (no) followed by Class 1 (yes) in 3 digits.
        """
        vals = self.df['y'].value_counts(normalize=True)
        return (round(vals.loc['no'],3), round(vals.loc['yes'],3))

    def Q4(self): # DO NOT modify this line
        """
        Problem 4:
            Remove duplicate records from the data. What are the shape of the dataset afterward?
        """
        self.df = self.df.drop_duplicates()
        return self.df.shape

    def Q5(self): # DO NOT modify this line
        """
        Problem 5:
            5. Replace unknown value with null
            6. Remove features with more than 99% flat values. 
                Hint: There is only one feature should be drop
            7. Split Data
            -	Split the dataset into training and testing sets with a 70:30 ratio.
            -	random_state=0
            -	stratify option
            return the tuple of shapes of X_train and X_test.

        """
        self.Q4()
        class dropFlat(BaseEstimator, TransformerMixin):
            def __init__(self, threshold=0.99):
                self.threshold = threshold
                self.keep_col = []
            def fit(self, X, y=None):
                self.keep_col = []
                for col in X.columns.tolist():
                    if X[col].value_counts(normalize=True).iloc[0] <= 0.99:
                        self.keep_col.append(col)
                return self
            def transform(self, X):
                return X[self.keep_col]
            def fit_transform(self, X, y = None, **fit_params):
                self.fit(X, y)
                return self.transform(X)
            def get_feature_names_out(self):
                return self.keep_col
        
        fn = FunctionTransformer(lambda X: X.replace('unknown', np.nan))
        cols = self.df.columns
        self.df = pd.DataFrame(fn.fit_transform(self.df), columns=cols)
        drop_flat = dropFlat(threshold=0.99)
        self.df = pd.DataFrame(drop_flat.fit_transform(self.df), columns=drop_flat.get_feature_names_out())
        y = self.df.pop("y")
        x = self.df.copy()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x,y, stratify=y, random_state=0, test_size=0.3)
        return (self.X_train.shape, self.X_test.shape)

    def Q6(self): 
        """
        Problem 6: 
            8. Impute missing
                -	For numeric variables: Impute missing values using the mean.
                -	For categorical variables: Impute missing values using the mode.
                Hint: Use statistics calculated from the training dataset to avoid data leakage.
            9. Categorical Encoder:
                Map the nominal data for the education variable using the following order:
                education_order = {
                    'illiterate': 1,
                    'basic.4y': 2,
                    'basic.6y': 3,
                    'basic.9y': 4,
                    'high.school': 5,
                    'professional.course': 6,
                    'university.degree': 7} 
                Hint: Use One hot encoder or pd.dummy to encode nominal category
            return the shape of X_train.

        """
        self.Q5()
        cate_cols = self.X_train.select_dtypes(include='object').columns.tolist()
        num_cols = self.X_train.select_dtypes(include='number').columns.tolist()
        cate_cols.remove('education')
        edupipe = make_pipeline(
            (SimpleImputer(strategy="most_frequent")),
            (OrdinalEncoder(categories=[[
                    'illiterate',
                    'basic.4y',
                    'basic.6y',
                    'basic.9y',
                    'high.school',
                    'professional.course',
                    'university.degree']])),
        )
        nom_pipe = make_pipeline(
            (SimpleImputer(strategy="most_frequent")),
            (OneHotEncoder(sparse_output=False))
        )
        pipe = make_column_transformer(
            (SimpleImputer(strategy="mean"), num_cols),
            (nom_pipe, cate_cols),
            (edupipe, ['education']),
        )
        pipe.set_output(transform='default')
        self.X_train = pipe.fit_transform(self.X_train)
        self.X_test = pipe.transform(self.X_test)
        return self.X_train.shape
    
    def Q7(self):
        ''' Problem7: Use Logistic Regression as the model with 
            random_state=2025, 
            class_weight='balanced' and 
            max_iter=500. 
            Train the model using all the remaining available variables. 
            What is the macro F1 score of the model on the test data? in 3 digits
        '''
        self.Q6()
        model = LogisticRegression(random_state=2025, class_weight='balanced', max_iter=500)
        pipe = make_pipeline(
            model
        )
        pipe.fit(self.X_train, self.y_train)
        predict = pipe.predict(self.X_test)
        report = classification_report(self.y_test, predict, output_dict=True)
        return round(report['macro avg']['f1-score'],2)