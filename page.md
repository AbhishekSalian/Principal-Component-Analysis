
**Created by Abhishek C. Salian**

$\Huge {\underline{PRINCIPAL \,\,\,\,\,COMPONENT\,\,\,\, \, ANALYSIS \,\,\,\,\, (PCA)}}$

#### What is need of PCA ?


PCA is done to find a lower dimensional plane (i.e Surface) to project the data such that the sum of least squares of point from that plane is minimum.

The following steps will help to understand algorithm for PCA:-

## STEP 1:- Standardize the data to obtain $X_s$ .

 We standardize features by **removing the mean and scaling to unit variance**
To do this following is the **vectorized** implementation. 


$\Large X_s =\Bigl(\frac{X - \mu}{\sigma}\Bigr)$ 

$\Large \mu = \frac{1}{N}\sum_{i=1}^N(x_i)$

$\Large \sigma =\sqrt{\frac{1}{N}\sum_{i=1}^N(x_i - \mu)^2}$

where $X_s$  is standardized feature vectors ; **X** is  feature vectors ;  $\mu$ is means ; $\sigma$ is standard deviation.

                    

For 2 feature vector i.e $X \in R^2$  following is standardization result

![image.png](attachment:image.png)

    

For 3 feature vector i.e *$X \in R^3 $* following is standardization result

![image.png](attachment:image.png)

## STEP 2 :- Eigen decomposition of $X_s$

We can find the eigen vectors and eigen values of covariance matrix of $X_s$ using the below formula:  

$\Large Covariance\,\,\,matrix =  S =$  $\Large \left[ {\begin{array}{cc}
                           \sigma_{xx}^2  &  \sigma_{xy}\\
                            \sigma_{yx}  &  \sigma_{yy}^2\\
                        \end{array} } \right]$

Formula for each element of can be written as

![image.png](attachment:image.png)

Note:-Similar matrix can be computed for N dimensional data

### Further we calculate the eigen values and eigen vectors using: Det(S - $\lambda.I$)=0

### After finding eigen values we can compute eigen vectors for corresponding eigen values using 
$\large (S - \lambda \cdot I)X\,\,=\,\,\, 0$ 

## OR we can use SVD method for the same task

When we decompose the data or signal we get the  characteristic vectors of that data or signals which we call as **Eigen Vectors**.
Eigen vectors donot get rotated they can only be stretched.
Now this stretching Factor for **Eigen Vectors** we call it as **Eigen Values**

### Every M X N matrix factorizes into   

$\Large A = U\sum V^T$ 

### Where U is Eigen vector and  $\sum$ is Eigen value, which we will need in futher steps

### U is an (m x m) orthogonal matrix, 

### 𝚺 is an (m x n) nonnegative rectangular diagonal matrix

### V is an (n x n) orthogonal matrix

Below are the steps  for eigen decomposition using svd

1) Compute its transpose $A^TA$

2) Determine the eigenvalues of $A^TA$ and sort these in descending order, in the absolute
   sense. Square roots these to obtain the singular values of A.

3) Construct diagonal matrix S by placing singular values in descending order along its
   diagonal. Compute its inverse, $S^{-1}$

4) Use the ordered eigenvalues from step 2 and compute the eigenvectors of $A^TA$. Place these eigenvectors along the columns of V and compute its transpose,
    $V^T$

5) Compute U as $ U \,\,=\,\, AVS^{-1}$. To complete the proof, compute the full SVD using $A_r\,\, =\,\, USV^T$ for verification 
   such that $A_r\,\,=\,\,A$

This can be directly computed using scipy's linear algebra package  

Resulting Pricipal components would look like below figure

![image.png](attachment:image.png)

## STEP 3 :-Sorting and Selecting Principal Components Eigen values

We sort  eigen values in decreasing  order .Simultaneous ordering of eigen vectors should be done with eigen values

Out of "n" dimension we will only select "k" dimension where k<n.(this is where we are reducing dimension by only selecting **Principal Components).

![image.png](attachment:image.png)

                                          figure 1                                               figure 2

Above figures consist of two explained variance plot of two distribution of datas.This shows us the variance associated with each eigen values

#### for figure 1:Eigen values variance: [71.92351700769406, 28.07648299230595]

#### for figure 2:Eigen values variance: [86.15729793420535, 11.466302603287813, 2.376399462506849] 

**More is the magnitude  of eigen value more is the variance associated with it** .Therefore, we will select only those eigen values with high variance since maximum information is associated with it 

for e.g

$\lambda_1=0.76 ,\lambda_2=0.14 , \lambda_3=0.08,\lambda_4=0.02$

The above e.g is 4D data's eigen values this can be reduced to 3D by selecting only $\lambda_1, \lambda_2,\lambda_3,$ with total variance of 98%

# STEP 4:-Construct the projection matrix U from the selected k eigenvectors.

Create a **Projection matrix U_proj** which will formed by stacking the **Eigen Vectors** corresponding to selected "K" lambdas
i.e **Principal Components**

 $\LARGE U =\left[ {\begin{array}{cc}
   |  &  |    .... &  |\\
  u^1 & u^2  .... & u^n\\
   |  &  |    .... &  | \\
  \end{array} } \right]$

$\Huge            \downarrow$

$\LARGE U_{proj}=\left[ {\begin{array}{cc}
   |  &  |   .... &  |\\
  u^1 & u^2  .... & u^k\\
   |  &  |   .... &  | \\
  \end{array} } \right]$

# STEP 5:-Transforming the original dataset via U to obtain a k -dimensional feature subspace.

Take Hadamard product between $U_{proj}$ and **$X_s$**

$\huge X_{pca} = X_s \cdot U_{proj}^T $

### This $X_{pca}$ can be used to train a model

### Take inverse transform of $X_{pca}$ using  $X_{pca} + \mu$

### Therefore,reconstructed data becomes  $X_{reconst} = X_{pca} +\mu$

(for data1)

![image.png](attachment:image.png)
