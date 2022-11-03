# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 14:03:56 2020

@author: tyler
"""

import numpy as np
import pandas as pd


heartdisease = pd.read_csv("C:/Users/tyler/OneDrive/Documents/school2k20/Fall2020/CIS3339/HW6/archive/framingham_heart_disease.csv", sep= ',') 

heartdisease.dropna()

np.savetxt("C:/Users/tyler/OneDrive/Documents/school2k20/Fall2020/CIS3339/HW6/framingham_heart_disease.csv", heartdisease, delimiter=",")
=