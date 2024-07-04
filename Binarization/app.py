class Solution:
     def __init__(self, df):
          self.df = df
          
     def Binarization(self):
          import numpy as np
          import pandas as pd
          from sklearn import preprocessing
          Input_Array = self.df[['Home', 'Offers']].values
          Data_Binary = preprocessing.Binarizer(threshold=1).transform(Input_Array)
          return Data_Binary
          
     def Mean_Removal(self):
          from sklearn import preprocessing
          import numpy as np
          Input_Array = self.df[['Home', 'Offers']].values
          print("Mean Value : ", Input_Array.mean(axis = 0))
          print("STD = ", Input_Array.std(axis = 0))
          Scaled_Mean = preprocessing.scale(Input_Array)
          print("Scaled Mean = ", Scaled_Mean.mean(axis = 0))
          print("Scaled STD = ", Scaled_Mean.std(axis = 0))
          
     def Min_Max_Scalar(self):
          from sklearn import preprocessing
          Input_Array = self.df[['Home', 'Offers']].values
          data_scalar_minmax = preprocessing.MinMaxScaler(feature_range=(0,1))
          data_scaled_minmax = data_scalar_minmax.fit_transform(Input_Array)
          return f"MinMax Scalar is: {data_scaled_minmax}"