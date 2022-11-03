# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 11:34:52 2020

@author: tyler
"""
import pandas as pd
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import statistics
import seaborn as sns

file1=pd.read_csv("C:\Users\tyler\OneDrive\Documents\SCHOOL\school2k20\Fall2020\CIS3339\Hw1\titanic.txt",sep=",")


file1.shape
file1.head()
file1.columns


plt.boxplot(file1['age'])
file1['age'].describe()
IQR=np.nanpercentile(file1['age'],75)-np.nanpercentile(file1['age'],25)
UpperBound=np.nanpercentile(file1['age'],75)+1.5*IQR
LowerBound=np.nanpercentile(file1['age'],25)+1.5*IQR
file1['age']>UpperBound

plt.boxplot(file1['age'])
plt.hlines(y=UpperBound,xmin=0,xmax=2,colors='red')

howmanyoutliers=sum(file1['age'])>UpperBound
