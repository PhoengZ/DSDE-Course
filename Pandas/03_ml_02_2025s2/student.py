
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import warnings # DO NOT modify this line
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
        return (len(self.df.select_dtypes(include="number").columns),len(self.df.select_dtypes(include="object").columns)) 
    
    def Q3(self): # DO NOT modify this line
        """
        Problem 3:
            return the tuple of the Class 0 (no) followed by Class 1 (yes) in 3 digits.
        """
        vals = self.df["y"].value_counts(normalize=True).values
        return (round(vals[0],3), round(vals[1],3))
      
    

    def Q4(self): # DO NOT modify this line
        """
        Problem 4:
            Remove duplicate records from the data. What are the shape of the dataset afterward?
        """
        self.df.drop_duplicates(inplace=True)
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
        self.df.replace("unknown", np.nan, inplace=True)
        cols = self.df.columns
        for col in cols:
            if self.df[col].value_counts(normalize=True).iloc[0] > 0.99:
                self.df.drop([col], axis=1, inplace=True)
        y = self.df['y']
        x = self.df.drop('y', axis=1)
        x_train, x_test, y_train, y_test = train_test_split(x,y, stratify=y, random_state=0, test_size=0.3)
        self.X_train, self.X_test, self.y_train, self.y_test = x_train, x_test, y_train, y_test
        return (x_train.shape, x_test.shape)
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
        education_order = {
            'illiterate': 1,
            'basic.4y': 2,
            'basic.6y': 3,
            'basic.9y': 4,
            'high.school': 5,
            'professional.course': 6,
            'university.degree': 7
        } 
        self.Q5()
        num_cols = self.X_train.select_dtypes(include='number').columns.tolist()
        cate_cols = self.X_train.select_dtypes(include='object').columns.tolist()
        imp_num = SimpleImputer(strategy='mean')
        imp_cate = SimpleImputer(strategy='most_frequent')
        one_enc = OneHotEncoder(sparse_output=False)
        self.X_train[num_cols] = imp_num.fit_transform(self.X_train[num_cols])
        self.X_train[cate_cols] = imp_cate.fit_transform(self.X_train[cate_cols])
        self.X_test[num_cols] = imp_num.transform(self.X_test[num_cols])
        self.X_test[cate_cols] = imp_cate.transform(self.X_test[cate_cols])
        cate_cols.remove('education')
        one_cols_train = one_enc.fit_transform(self.X_train[cate_cols])
        one_cols_test = one_enc.transform(self.X_test[cate_cols])
        one_cols_train = pd.DataFrame(one_cols_train, columns=one_enc.get_feature_names_out(), index=self.X_train.index)
        one_cols_test = pd.DataFrame(one_cols_test, columns=one_enc.get_feature_names_out(), index=self.X_test.index)
        self.X_train.drop(cate_cols, axis=1, inplace=True)
        self.X_test.drop(cate_cols, axis=1, inplace=True)
        self.X_train = pd.concat([self.X_train, one_cols_train], axis=1)
        self.X_test = pd.concat([self.X_test, one_cols_test], axis=1)
        self.X_train["education"] = self.X_train["education"].map(education_order)
        self.X_test["education"] = self.X_test["education"].map(education_order)
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
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        report = classification_report(self.y_test, y_pred, output_dict=True)
        return round(report['macro avg']['f1-score'],2)

