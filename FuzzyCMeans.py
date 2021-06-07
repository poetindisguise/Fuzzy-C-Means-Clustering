import numpy as np, numpy.random
from scipy.spatial import distance

from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import davies_bouldin_score

from google.colab import drive
import pandas as pd

drive.mount('/content/drive')

from csv import writer
def add_row(fname, elements_list):                  #adding a new row containing silhoutte and DB Index values for a project
    with open(fname, 'a+', newline='') as write_object:
        csv_writer = writer(write_object)
        csv_writer.writerow(elements_list)

np.seterr(divide='ignore', invalid='ignore')

m=2

for t in range(1,57):                               #loop for project files 1-56
    var_db=list()                                   #contains list of DBIndex values for a particular project
    var_silh=list()                                 #contains list of silhouette coeff values for a particular project
    for c in range(2,11):                           #from cluster no. 2 to 10
        file_str = "/content/drive/MyDrive/data_fcm/"+str(t)+".csv"

        X=np.genfromtxt(file_str,delimiter=',')     #X is a matrix of values from t.csv
        X=np.delete(X, -1,1)                        #removing last column from matrix
        d=len(X[0])                                 #no. of features (dimensions)
        total_rows = len(X)                         #no. of rows

        extraZeros = np.zeros((total_rows, 1))
        X = np.append(X, extraZeros, axis=1)        #last column with zeros added to X. Later on, cluster_number will be fed in this column.

        centroids_array = np.zeros((c,d+1))         #empty array of centroids. c rows,d+1 columns
  
        # Randomly initialize the weight matrix
        wt_arr = np.random.dirichlet(np.ones(c),size=total_rows) #random Dirichlet distribution (sum = 1)

        for it in range(300):                               # total number of iterations = 300
            # Computing new centroids
            for j in range(c):                              
                                                            #wt[i][j] means weight of ith data in jth cluster            
                sum_of_num =0
                for i in range(total_rows):           
                    temp_sum = np.multiply(np.power(wt_arr[i,j],m),X[i,:])      #sum of ((weight^m)*x)
                    sum_of_num +=temp_sum  

                sum_of_denom = sum(np.power(wt_arr[:,j],m))      #sum of (weight^m)
                
                new_cent = sum_of_num/sum_of_denom
                centroids_array[j] = np.reshape(new_cent,d+1)   

            # Updating the fuzzy pseudo partition
            for i in range(total_rows):
                temp_sum = 0
                for j in range(c):                      #formula of membership-value calculation used
                    temp_sum += np.power(1/distance.euclidean(centroids_array[j,0:d],X[i,0:d]),2/(m-1))     
                for j in range(c):
                    new_wt = np.power((1/distance.euclidean(centroids_array[j,0:d],X[i,0:d])),2/(m-1))/temp_sum    
                    wt_arr[i,j] = new_wt  

        for i in range(total_rows):    
            cluster_num = np.where(wt_arr[i] == np.amax(wt_arr[i]))    #placing the element in the cluster for which its weight is max.
            X[i,d] = cluster_num[0]

        temp=(X[:, d])      #last column of X matrix (which contains cluster numbers)
        
        var_db.append(davies_bouldin_score(X,temp))     #adding DBindex value in the list var_db
  
        var_silh.append(silhouette_score(X,temp))       #adding silhouette coeff value in the list var_silh

    final_list=list()
    final_list = [t] + var_silh + var_db                #t.csv + silh values + db_values

    res = '/content/drive/MyDrive/data_fcm/result.csv'

    add_row(res,final_list)                             #adding final list as a new row in the result file

