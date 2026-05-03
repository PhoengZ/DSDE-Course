#import your other libraries here
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
import numpy as np
class Clustering:
    def __init__(self, file_path): # DO NOT modify this line
        #Add other parameters if needed
        self.file_path = file_path 
        self.df = None #parameter for loading csv

    def Q1(self): # DO NOT modify this line
        """
        Step1-4
            1. Load the CSV file.
            2. Choose edible mushrooms only.
            3. Only the variables below have been selected to describe the distinctive
               characteristics of edible mushrooms:
               'cap-color-rate','stalk-color-above-ring-rate'
            4. Provide a proper data preprocessing as follows:
                - Fill missing with mean
                - Standardize variables with Standard Scaler
        """
        self.df = pd.read_csv(self.file_path)
        self.df = self.df.loc[self.df['label'] == 'e', :]
        self.df = self.df.loc[:, ['cap-color-rate','stalk-color-above-ring-rate']]
        self.pipeline = make_pipeline(
            (SimpleImputer(strategy="mean")),
            (StandardScaler())
        )
        self.pipeline.set_output(transform="pandas")
        self.df = self.pipeline.fit_transform(self.df)
        return self.df.shape

    def Q2(self): # DO NOT modify this line
        """
        Step5-6
            5. K-means clustering with 5 clusters (n_clusters=5, random_state=0, n_init='auto')
            6. Show the maximum centroid of 2 features ('cap-color-rate' and 'stalk-color-above-ring-rate') in 2 digits.
        """
        self.Q1()
        self.Km = KMeans(n_clusters=5, random_state=0, n_init="auto")
        self.Km.fit(self.df)
        m = np.max(self.Km.cluster_centers_, axis=0)
        return m.round(2)

    def Q3(self): # DO NOT modify this line
        """
        Step7
            7. Convert the centroid value to the original scale, and show the minimum centroid of 2 features in 2 digits.

        """
        self.Q2()
        mm = self.Km.cluster_centers_
        mm = np.min(self.pipeline[1].inverse_transform(mm), axis=0)
        return mm.round(2)
