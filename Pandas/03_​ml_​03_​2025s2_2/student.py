import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

class Clustering:
    def __init__(self, file_path): # DO NOT modify this line
        #Add other parameters if needed
        self.file_path = file_path 
        self.df = pd.read_csv(self.file_path) #parameter for loading csv

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
        self.df = self.df.loc[self.df['label'] == 'e', :]
        self.df = self.df.loc[:, ['cap-color-rate','stalk-color-above-ring-rate']]
        self.df = self.df.fillna(self.df.mean(numeric_only=True))
        self.sca = StandardScaler()
        self.df = pd.DataFrame(self.sca.fit_transform(self.df), columns=self.df.columns.tolist(), index=self.df.index)
        return self.df.shape
        
    def Q2(self): # DO NOT modify this line
        """
        Step5-6
            5. K-means clustering with 5 clusters (n_clusters=5, random_state=0, n_init='auto')
            6. Show the maximum centroid of 2 features ('cap-color-rate' and 'stalk-color-above-ring-rate') in 2 digits.
        """
        self.Q1()
        km = KMeans(n_clusters=5, random_state=0, n_init='auto')
        results = km.fit(self.df)
        self.km = km
        ans = np.max(results.cluster_centers_, axis=0)
        return (round(ans[0],2), round(ans[1],2))

    def Q3(self): # DO NOT modify this line
        """
        Step7
            7. Convert the centroid value to the original scale, and show the minimum centroid of 2 features in 2 digits.

        """
        self.Q2()
        centroids = self.sca.inverse_transform(self.km.cluster_centers_)
        ans = np.min(centroids, axis=0)
        return (round(ans[0],2), round(ans[1],2))

               