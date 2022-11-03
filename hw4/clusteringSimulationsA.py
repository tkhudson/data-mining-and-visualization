# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 08:44:47 2019

@author: rm84
"""
import pandas as pd
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
from scipy.spatial.distance import mahalanobis
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sea
from seaborn import distplot
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.mixture import GaussianMixture
from scipy import random, linalg
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D 

num_samples = 400

# The desired mean values of the sample.
mu = np.array([5.0, 0.0, 10.0])

# The desired covariance matrix.
r = np.array([
        [  3.40, -2.75, -2.00],
        [ -2.75,  5.50,  1.50],
        [ -2.00,  1.50,  1.25]
    ])

# Generate the random samples.
y = np.random.multivariate_normal(mu, r, size=num_samples)


#also for cov matrix: np.cov(np.transpose(X))
Sinv=inv(np.cov(y[:,0],y[:,2]))
X=np.zeros((400,2))
X[:,0]=y[:,0]
X[:,1]=y[:,2]
M=[X[:,0].mean(),X[:,1].mean()]
#need the diagonal
G=np.sqrt(np.diag(np.matmul(np.matmul(X-M,Sinv),(X-M).transpose(1,0))))
#######################################3
A=np.array([5.17372027, 9.67460694]).reshape(2,1)
AT=A.transpose()
B=np.array([3, 11]).reshape(2,1)
BT=B.transpose()
M=np.array(M).reshape(2,1)
MT=M.transpose()
FirstMult=np.matmul((AT-MT),Sinv)
SecondMult=np.matmul(FirstMult,(A-M))
Result=np.sqrt(SecondMult)
#another way


def mahalanobisR(z,meanCol,IC):
    m = []
    for i in range(z.shape[0]):
        m.append(mahalanobis(z[i,:],meanCol,IC))
    return(m)

mR = mahalanobisR(X,MT,Sinv)
BR = mahalanobisR(BT,MT,Sinv)

# Plot various projections .
plt.subplot(2,2,3)
plt.plot(y[:,0], y[:,2], 'b.')
plt.plot(mu[0], mu[2], 'ro')
plt.xlabel('X_1')
plt.ylabel('X_2')
plt.axis('equal')
plt.grid(True)
X=[6,3]
Y=[12,11]
plt.plot(X, Y, 'go',markersize=10)
plt.annotate('sqrt((6-5)^2+abs(12-10)^2)=sqrt(5)', xy=(6,11), xytext=(6,11)
            )
plt.annotate('sqrt((3-5)^2+(11-10)^2)=sqrt(5)', xy=(-0.25,10.5), xytext=(-.25,10.5)
            )
y_number_values=[10,12,10,11]
x_number_values=[5,6,5,3]
plt.plot((x_number_values)[0:2],(y_number_values)[0:2], 'black',linewidth=3)
plt.plot((x_number_values)[2:4],(y_number_values)[2:4], 'black',linewidth=3)

###############plot the Mahalonobic normalized distances and show the above...

#######################################################################
#######################################################################
#Simulated Case 1
#Generate uncorrelated data with 2 populations 
num_samples = 5000

# The desired mean values of the sample.
mu1 = np.array([5.0, 25.0])

# The desired covariance matrix.
r1 = np.array([
        [  3, 0],
        [ 0,  5]])

# Generate the random samples.
y1 = np.random.multivariate_normal(mu1, r1, size=num_samples)

mu2 = np.array([30, 5.0])

# The desired covariance matrix.
r2 = np.array([
        [  2, 0],
        [ 0,  3]])

# Generate the random samples.
y2 = np.random.multivariate_normal(mu2, r2, size=num_samples)
y=np.vstack((y1,y2))
plt.scatter(y[:,0],y[:,1])
kmeans = KMeans(n_clusters=2) 
scaler = MinMaxScaler()
y_scaled = scaler.fit_transform(y)
kmeans.fit(y_scaled)
plt.scatter(y_scaled[:,0],y_scaled[:,1], c=kmeans.labels_, cmap='rainbow')
(kmeans.cluster_centers_)

kmeans = KMeans(n_clusters=3) 
scaler = MinMaxScaler()
y_scaled = scaler.fit_transform(y)
kmeans.fit(y_scaled)
(kmeans.cluster_centers_)
plt.scatter(y_scaled[:,0],y_scaled[:,1], c=kmeans.labels_, cmap='rainbow')

kmeans = KMeans(n_clusters=4) 
scaler = MinMaxScaler()
y_scaled = scaler.fit_transform(y)
kmeans.fit(y_scaled)
(kmeans.cluster_centers_)
plt.scatter(y_scaled[:,0],y_scaled[:,1], c=kmeans.labels_, cmap='rainbow')

###This can be shown as something to be aware of. Problematic
##################################################################### 
sse={}
silhavg={}
for k in range(1,10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(y_scaled)
    cluster_labels = kmeans.fit_predict(y_scaled)
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center

for k in range(2,10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(y_scaled)
    cluster_labels = kmeans.fit_predict(y_scaled)
    silhavg[k] = silhouette_score(y_scaled, cluster_labels)
    
plt.figure()
plt.subplot(1,2,1)
plt.plot(list(sse.keys()), list(sse.values()))
plt.title('SSE Measure')
plt.subplot(1,2,2)
plt.plot(list(range(2,10)), list(silhavg.values()))
plt.title('Avg Silhoutte Scores')
#############################################################################3

############################################################################
U=np.zeros((file4.shape[0],2))
U[:,0]=file4.iloc[:,1]
U[:,1]=file4.iloc[:,3]
scaler = MinMaxScaler()


kmeans = KMeans(n_clusters=4) 
X_scaled = scaler.fit_transform(U)
plt.scatter(x=X_scaled[:,0],y=X_scaled[:,1])

kmeans.fit(X_scaled)
(kmeans.cluster_centers_)
(kmeans.labels)
(kmeans.labels_)[0:10]

plt.scatter(X_scaled[:,0],X_scaled[:,1], c=kmeans.labels_, cmap='rainbow')

#####
#symmentric + semi defn matrix

matrixSize = 2 
A = random.normal(3,3,4).reshape(matrixSize,matrixSize)
B = np.dot(A,A.transpose())

A = random.normal(3,3,4).reshape(matrixSize,matrixSize)
C = np.dot(A,A.transpose())

#######Correlated Data
num_samples = 5000

# The desired mean values of the sample.
mu1 = np.array([10.0, 20.0])

# The desired covariance matrix.
r1 = B

# Generate the random samples.
y1 = np.random.multivariate_normal(mu1, r1, size=num_samples)

mu2 = np.array([20, 15.0])

# The desired covariance matrix.
r2 = C

# Generate the random samples.
y2 = np.random.multivariate_normal(mu2, r2, size=num_samples)
y=np.vstack((y1,y2))
#np.savetxt("C:/Users/rm84/Desktop/Teaching/3339/Cluster Analysis/2highcorrpop.csv", y, delimiter=",")
plt.scatter(x=y[:,0],y=y[:,1])

kmeans = KMeans(n_clusters=4) 
scaler = MinMaxScaler()
y_scaled = scaler.fit_transform(y)
kmeans.fit(y_scaled)
(kmeans.cluster_centers_)
plt.scatter(y_scaled[:,0],y_scaled[:,1], c=kmeans.labels_, cmap='rainbow')

#np.savetxt("C:/Users/rm84/Desktop/Teaching/3339/Cluster Analysis/2highcorrpopwellsplit.csv", y, delimiter=",")
#############################################################################3
###This can be shown as something to be aware of. Problematic

cluster_labels=kmeans.labels_


sse={}
silhavg={}
for k in range(1,10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(y_scaled)
    cluster_labels = kmeans.fit_predict(y_scaled)
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center

for k in range(2,10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(y_scaled)
    cluster_labels = kmeans.fit_predict(y_scaled)
    silhavg[k] = silhouette_score(y_scaled, cluster_labels)



plt.figure()
plt.subplot(1,2,1)
plt.plot(list(sse.keys()), list(sse.values()))
plt.title('SSE Measure')
plt.subplot(1,2,2)
plt.plot(list(range(2,10)), list(silhavg.values()))
plt.title('Avg Silhoutte Scores')

 

#################################################################

####Even though the Mahalanobis distance 
####
plt.subplot(2,2,1)
plt.scatter(y[:,0],y[:,1], cmap='rainbow')
plt.title('Raw Data 2 Populations')

kmeans = KMeans(n_clusters=2,n_init=300,init='k-means++') 
kmeans.fit(y_scaled)
plt.subplot(2,2,2)
plt.scatter(y[:,0],y[:,1], c=kmeans.labels_, cmap='rainbow')
plt.title('K=2 Euclidean')
#reshape necc for clustering
kmeans = KMeans(n_clusters=2,n_init=300,init='k-means++') 
kmeans.fit(np.array(mR).reshape(-1,1))
plt.subplot(2,2,3)
plt.scatter(y[:,0],y[:,1], c=kmeans.labels_, cmap='rainbow')
plt.title('K=2 Mahalanobis')

np.unique(kmeans.labels_,return_counts=True)


gmm = GaussianMixture(n_components=2).fit(y_scaled)
labels = gmm.predict(y_scaled)
plt.subplot(2,2,4)
plt.scatter(y[:, 0], y[:, 1], c=labels,cmap='rainbow');
plt.title('K=2 Probabilistic Modelling')

scaler = MinMaxScaler()
y_scaled = scaler.fit_transform(y)
###########################################################################
sse={}
silhavg={}
for k in range(1,10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(y_scaled)
    cluster_labels = kmeans.fit_predict(y_scaled)
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center

for k in range(2,10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(y_scaled)
    cluster_labels = kmeans.fit_predict(y_scaled)
    silhavg[k] = silhouette_score(y_scaled, cluster_labels)
    
plt.figure()
plt.subplot(1,2,1)
plt.plot(list(sse.keys()), list(sse.values()))
plt.title('SSE Measure')
plt.subplot(1,2,2)
plt.plot(list(range(2,10)), list(silhavg.values()))
plt.title('Avg Silhoutte Scores')
#############################################################################
AIC={}
silhavg={}
for k in range(1,10):
    gmm = GaussianMixture(n_components=k).fit(y_scaled)
    labels = gmm.predict(y_scaled)
    AIC[k] = gmm.aic(y_scaled) # Inertia: Sum of distances of samples to their closest cluster center

for k in range(2,10):
    gmm = GaussianMixture(n_components=k).fit(y_scaled)
    labels = gmm.predict(y_scaled)
    silhavg[k] = silhouette_score(y_scaled, labels)
    
plt.figure()
plt.subplot(1,2,1)
plt.plot(list(AIC.keys()), list(AIC.values()))
plt.title('AIC Measure')
plt.subplot(1,2,2)
plt.plot(list(range(2,10)), list(silhavg.values()))
plt.title('Avg Silhoutte Scores')

countries = pd.read_csv("C:/Users/tyler/OneDrive/Documents/school2k20/Fall2020/CIS3339/Data/countries of the world.csv", sep= ',') 

y=countries.iloc[:,2:]

ynonmiss=y.dropna()

kmeans = KMeans(n_clusters=2) 

scaler = MinMaxScaler()

y_scaled = scaler.fit_transform(ynonmiss)

kmeans.fit(y_scaled)

(kmeans.cluster_centers_)

sse={}

for k in range(1,15):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(y_scaled)
    cluster_labels = kmeans.fit_predict(y_scaled)
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center

plt.plot(list(sse.keys()), list(sse.values()))
plt.title('SSE Measure')









































#####################################################################
