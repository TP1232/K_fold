import numpy as np
import random
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
# from sklearn.datasets import make_classification

# =============================================================================
# ------------------------- Dataset load ------------------------------------

df = pd.read_csv("data/messidor_features.arff", header = None)
data = df.to_numpy()

# =============================================================================
# global Variable for all functions
global acc_scores
global n

# Calculating total number of features
n = len(df.count()) - 1

# =============================================================================
#
def normalize(x):

## normalization function for min max
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x-min_val) / (max_val-min_val)
    return x

# =============================================================================

# data_norm = normalize(data)

# =============================================================================
# 
def data_prep(data,k,fold_count):
    
    test = np.array(data[k])
    arr = np.arange(fold_count).tolist()
    arr.remove(k)
    train = data[arr[0]]
    for i in range(1,len(arr)):
        train = np.concatenate((train,data[arr[i]]))
    test = np.array(test)
    X_train,Y_train,X_test,Y_test = normalize(train[:,:-1]) ,train[:,:-1], normalize(test[:,:-1]) , test[:,:-1]

    return ([X_train,X_test,Y_train,Y_test])

# =============================================================================

def cross_val_fold(data,fold_count):
    
# divideng data set into equal fold to perform crossfold validation
    classes = np.unique(data[:,19])
    n_class = len(classes)
    data_fold,elements_for_fold = [], []
    
    for i in classes:
        current_class = data[np.where(data[:,19]==i)]
        random.shuffle(current_class)
        data_fold.append(current_class)
        current_class_elements = int(len(current_class)/fold_count)
        elements_for_fold.append(current_class_elements)
    dataset_split = []
    for i in range(fold_count):
        fold = []
        for j in range(n_class):
            # n_samp = elements_for_fold[j]
            
            if(i!=fold_count -1):
                fold.extend(data_fold[j][i*elements_for_fold[j]:(i+1)*elements_for_fold[j]])
            else:
                fold.extend(data_fold[j][i*elements_for_fold[j]:])
                
        random.shuffle(fold)
        dataset_split.append(fold)
    return dataset_split

# =============================================================================
def fit(data_parts,fold_counts):

    max_acc = 0.0
    data_folds = []
    
#  Expand the data from it's data parts
    for i in range(fold_counts):
        data_folds.append(data_prep(data_parts,i,fold_counts))
    feature,score = [],[]
#  Selecting any one feature randomy  for wrapper approch
    for i in range(n):
# Creating accuracy for selecting best features amoung the all 
        accuracy = []
        for j in range(fold_counts):
            [X_train,X_test,Y_train,Y_test] = data_folds[j]
 # Using randomforest regressor  
            clf = RandomForestRegressor(max_depth=2, random_state=15).fit(X_train[:,i].reshape(-1,1), Y_train)
            accuracy.append(clf.score(X_test[:,i].reshape(-1,1), Y_test))
            
        score.append(np.mean(accuracy))
        
    feature.append(np.argmax(score))
    acc_scores = []
 # Chossing the features till the 75% percent of fetures is not selected
    while(len(feature) < 0.75*n):
        score = []
        
        for i in range(n):
            if(i in feature):
                score.append(0)
                continue
            accuracy = []
# Storing attributes for feature selection
            attributes = feature.copy()
            attributes.append(i)
            
            for i in range(fold_counts):
            
                [X_train,X_test,Y_train,Y_test] = data_folds[j]
            
                clf = RandomForestRegressor().fit(X_train[:,attributes], Y_train)
                accuracy.append(clf.score(X_test[:,attributes], Y_test))
                
            score.append(np.mean(accuracy))
# Storing the best value of accuracy and best feature combination
        print("-"*50)
        print("Features selected for Calculating accuracy: ",feature)
        print("Predicted Score is : ",np.max(score))
        if np.max(score) > max_acc:
            max_acc = np.max(score)
            best_features = feature.copy()
            
        print('Best Feature combination is : {} with accuracy {:.2f}'.format(best_features,max_acc*100 ))
        print("="*50)
        acc_scores.append(np.max(score))
        
        feature.append(np.argmax(score))
    indx = np.argmax(acc_scores)
    
    return(feature[:indx+1])
#     
# =============================================================================
    
data_parts = cross_val_fold(data,5)
model = fit(data_parts,5)
            
            

    


 