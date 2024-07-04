from app import Solution
import pandas as pd
import numpy as np
from sklearn import preprocessing
df = pd.read_csv(r'C:\Users\mk744\OneDrive - Poornima University\Desktop\Feature Scaling\Data Files\house-prices.csv')
ans = Solution(df)
print("After Doing Binarization :")
print(ans.Binarization())
print(ans.Mean_Removal())
print(ans.Min_Max_Scalar())