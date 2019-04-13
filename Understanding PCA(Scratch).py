#!/usr/bin/env python
# coding: utf-8

# # Principal Component Analysis

#    1.PCA is a $statistical$  $method$.
# 
#    2.It lumps highly correlated variables together.

# **PCA Algorithm:**
# 1. Standardize the data to obtain $S$.
# 
# 2. Obtain the Eigenvectors and Eigenvalues of $S$ via the **Covariance Matrix** method or **Singular Vector Decomposition** (SVD) method.
# 
# 3. Sort eigenvalues in **descending** order and choose the $k$ eigenvectors that correspond to the $k$ largest eigenvalues where $k$ is the number of dimensions of the new feature subspace ($k \leq d$).
# 
# 4. Construct the projection matrix $U$ from the selected $k$ eigenvectors.
# 
# 5. Transform the original dataset via $U$ to obtain a $k$-dimensional feature subspace.

# To understand the concept lets take a sample dataset

# In[1]:


from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy import linalg
from matplotlib import style
style.use('ggplot')
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = loadmat('sample.mat')
X=data['X']
plt.scatter(X[:,0],X[:,1],marker="o",color="r")
plt.xlabel("X1")
plt.ylabel("X2")
plt.axhline(linewidth=6, color='b')
plt.axvline(linewidth=5, color='b')
plt.xlim(0,8),plt.ylim(0,8),plt.title("Without Standardization")
plt.show()


# In[3]:


from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import mpl_toolkits.mplot3d as plt3d
from scipy.io import loadmat
import numpy as np

np.random.seed(32) # random seed for consistency
data1= loadmat('C:\\Users\\Admin\\a) My datascience\\A)principal Component analysis\\Bird_small.mat')
X1=data1['A']
X1=X1[0]


fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
plt.rcParams['legend.fontsize'] = 10   
ax.plot(X1[:,0],X1[:,1],X1[:,2], 'o', markersize=8, color='red')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

ax.legend(loc='upper right')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
x=[*range(0,6)]
y=np.zeros((1,5))
z=[*range(0,3)]
ax.plot([*range(0,200)],list(np.zeros((1,200))[0]),list(np.zeros((1,200))[0]),color='b')
ax.plot(list(np.zeros((1,200))[0]),[*range(0,200)],list(np.zeros((1,200))[0]),color='orange')
ax.plot(list(np.zeros((1,200))[0]),list(np.zeros((1,200))[0]),[*range(0,200)],color='k')
plt.show()


# # Step 1:Standardize the data

# In[4]:


X_std = StandardScaler().fit_transform(X)
plt.scatter(X_std[:,0],X_std[:,1],marker="o",color="r")
plt.xlabel("X1_s"),plt.ylabel("X2_s"),plt.title("With Standardization")
plt.axhline(linewidth=1, color='b')
plt.axvline(linewidth=1, color='b')

plt.show()


# In[5]:


X_std1 = StandardScaler().fit_transform(X1)
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
plt.rcParams['legend.fontsize'] = 10   
ax.plot(X_std1[:,0],X_std1[:,1],X_std1[:,2], 'o', alpha=0.7,markersize=3, color='red')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.legend(loc='upper right')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
x=[*range(0,6)]
y=np.zeros((1,5))
z=[*range(0,3)]
ax.plot([*range(0,5)],list(np.zeros((1,5))[0]),list(np.zeros((1,5))[0]),color='b')
ax.plot(list(np.zeros((1,5))[0]),[*range(0,5)],list(np.zeros((1,5))[0]),color='orange')
ax.plot(list(np.zeros((1,5))[0]),list(np.zeros((1,5))[0]),[*range(0,5)],color='k')
plt.show()


# # Step2:Obtaining Eigen vectors and Eigen values using SVD.

# In[6]:


scaler = StandardScaler()
scaler.fit(X)
eigen_vecs, eigen_vals,V =linalg.svd(scaler.transform(X).T)
print("Eigen values:\n",eigen_vals,"\n")
print("Eigen vectors:\n",eigen_vecs)


# In[7]:


scaler1 = StandardScaler()
scaler1.fit(X1)
eigen_vecs1, eigen_vals1,V1 =linalg.svd(scaler1.transform(X1).T)
print("Eigen values:\n",eigen_vals1,"\n")
print("Eigen vectors:\n",eigen_vecs1)


# **More is the value of eigen value  higher is variance associated with the eigen value.Therfore 9.315 has more variance then 3.63**

# In[8]:


with plt.style.context('dark_background'):
    plt.figure(figsize=(5,9))
    plt.scatter(X_std[:,0], X_std[:,1], marker="o",s=30, edgecolors='r',facecolors='red')
    # setting aspect ratio to 'equal' in order to show orthogonality of principal components in the plot
    plt.gca().set_aspect('equal')
    plt.quiver(np.mean(X_std[:,0]),np.mean(X_std[:,1]), eigen_vecs[0,0],eigen_vecs[0,1], scale=eigen_vals[1],
               color='b',label="PC1")
    plt.quiver(np.mean(X_std[:,0]),np.mean(X_std[:,1]), eigen_vecs[1,0], eigen_vecs[1,1], scale=eigen_vals[0], color='g',
              label="PC2")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.axhline(linewidth=1, color='c')
    plt.axvline(linewidth=1, color='c')
    plt.legend()
    plt.grid()
    plt.show()


# In[9]:


fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
plt.rcParams['legend.fontsize'] = 10 
ax.plot(X_std1[:,0],X_std1[:,1],X_std1[:,2], 'o', alpha=0.7,markersize=3, color='red')
    # setting aspect ratio to 'equal' in order to show orthogonality of principal components in the plot
plt.gca().set_aspect('equal')
ax.quiver(np.mean(X_std1[:,0]),np.mean(X_std1[:,1]),np.mean(X_std1[:,2]), 
              eigen_vecs1[0,0],eigen_vecs1[0,1], eigen_vecs1[0,2],color='b',label="Eigenvector1")
    
ax.quiver(np.mean(X_std1[:,0]),np.mean(X_std1[:,1]),np.mean(X_std1[:,2]),
              eigen_vecs1[1,0], eigen_vecs1[1,1],eigen_vecs1[1,2], color='g',label="Eigenvector2")
    
ax.quiver(np.mean(X_std1[:,0]),np.mean(X_std1[:,1]),np.mean(X_std1[:,2]), 
              eigen_vecs1[2,0], eigen_vecs1[2,1],eigen_vecs1[2,2], color='yellow',label="Eigenvector3")
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.legend(loc='upper right')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
x=[*range(0,6)]
y=np.zeros((1,5))
z=[*range(0,3)]
ax.plot([*range(0,5)],list(np.zeros((1,5))[0]),list(np.zeros((1,5))[0]),color='cyan')
ax.plot(list(np.zeros((1,5))[0]),[*range(0,5)],list(np.zeros((1,5))[0]),color='orange')
ax.plot(list(np.zeros((1,5))[0]),list(np.zeros((1,5))[0]),[*range(0,5)],color='k')
plt.show()


# In[11]:


e1=eigen_vecs1[0]*eigen_vals[0]
e2=eigen_vecs1[1]*eigen_vals[1]
e3=eigen_vecs1[2]*eigen_vals1[2]

E=np.array([e1,e2,e3])
E[0,0]


# In[12]:


eigen_vecs1[:,0],eigen_vecs1[:,1], eigen_vecs1[:,2]


# In[13]:


fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
plt.rcParams['legend.fontsize'] = 10 
ax.plot(X_std1[:,0],X_std1[:,1],X_std1[:,2], 'o', alpha=0.7,markersize=3, color='red')
    # setting aspect ratio to 'equal' in order to show orthogonality of principal components in the plot
plt.gca().set_aspect('equal')
ax.plot([np.mean(X_std1[:,0]),E[0,0]],
        [np.mean(X_std1[:,0]),E[0,1]],
        [np.mean(X_std1[:,0]),E[0,2]],color='c',label="Eigenvector1")

ax.plot([np.mean(X_std1[:,0]),E[1,0]],
        [np.mean(X_std1[:,0]),E[1,1]],
        [np.mean(X_std1[:,0]),E[1,2]],color='g',label="Eigenvector1")
ax.plot([np.mean(X_std1[:,0]),E[2,0]],
        [np.mean(X_std1[:,0]),E[2,1]],
        [np.mean(X_std1[:,0]),E[2,2]],color='magenta',label="Eigenvector1")
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.legend(loc='upper right')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
x=[*range(0,6)]
y=np.zeros((1,5))
z=[*range(0,3)]
ax.plot([*range(0,5)],list(np.zeros((1,5))[0]),list(np.zeros((1,5))[0]),color='b')
ax.plot(list(np.zeros((1,5))[0]),[*range(0,5)],list(np.zeros((1,5))[0]),color='orange')
ax.plot(list(np.zeros((1,5))[0]),list(np.zeros((1,5))[0]),[*range(0,5)],color='k')
plt.show()


# # step3:Sorting and  Selecting Principal Components  Eigen values 

# In[14]:


eig_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])


# In[15]:


total = sum(eigen_vals)
var_exp = [(i / total)*100 for i in sorted(eigen_vals, reverse=True)]
print("Eigen values variance:",var_exp)

with plt.style.context('dark_background'):
    plt.figure(figsize=(6, 4))

    plt.bar(range(2), var_exp, alpha=0.7, align='center',color='cyan',label='individual explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()


# In[16]:


eig_pairs1 = [(np.abs(eigen_vals1[i]), eigen_vecs1[:,i]) for i in range(len(eigen_vals1))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs1.sort(key=lambda x: x[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs1:
    print(i[0])
total1 = sum(eigen_vals1)
var_exp1 = [(i / total1)*100 for i in sorted(eigen_vals1, reverse=True)]
print("Eigen values variance:",var_exp1)

with plt.style.context('dark_background'):
    plt.figure(figsize=(6, 4))

    plt.bar(range(3), var_exp1, alpha=0.7, align='center',color='cyan',label='individual explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()


# # step 4:Construct the projection matrix  U  from the selected  k  eigenvectors.

# In[17]:


U_proj = np.hstack((eig_pairs[0][1].reshape(2,1)))
print('Projection matrix U:\n',U_proj)


# In[18]:


(eig_pairs[0][1].reshape(2,1))


# In[19]:


U_proj1 = np.hstack((eig_pairs1[0][1].reshape(3,1),(eig_pairs1[1][1].reshape(3,1))))
print('Projection matrix U:\n',U_proj1)


# # Step 5:Transforming the original dataset via  U  to obtain a  k -dimensional feature subspace.

# In[20]:


X_pca= X_std.dot(U_proj)
print(X_pca.shape)
X_pca=X_pca.reshape(50,1)
print(X_pca.shape)

u=np.mean(X,axis=0)
u

X_rec=X_pca.dot(U_proj.reshape((1,2)))+u

X_rec[:3]

X[:3]


# In[21]:


X_pca1= X_std1.dot(U_proj1)

X_pca1.shape
u1=np.mean(X1,axis=0)


X_rec1=X_pca1.dot(U_proj1.reshape((2,3)))+u1

X_rec1[:3]

X1[:3]


# In[23]:


import seaborn as sns

plt.figure(figsize=(10,10))
plt.subplot(221)
sns.kdeplot(X[:,0], shade=True, color="r",label="X0_o")
plt.subplot(222)
sns.kdeplot(X[:,1], shade=True, color="b",label="X1_o")
plt.subplot(223)
sns.kdeplot(X_rec[:,0], shade=True, color="r",label="X0_rec")
plt.subplot(224)
sns.kdeplot(X_rec[:,1], shade=True, color="b",label="X1_rec")


# In[ ]:




