This is a scratch program (Without ML libraries) to implement the sequential forward selection algorithm from scratch to identify important features in the dataset.
## Feature Selection

- Feature selection is extremely important in machine learning primarily because it serves as a fundamental technique to direct the use of variables to what is most efficient and effective for a given machine learning system.

## K Fold Cross-Validation

- Cross-Validation can help to estimate the performance of the model. And K-Fold Cross Validation is One type of cross-validation. Here in this project 

## Abstract

Program to implement the sequential forward selection algorithm to identify important
features from scratch (without libraries). 
- The program will take one input: a dataset (20 features) where the last column
is the class variable. 
- The program will load the dataset, and then use a wrapper approach with
a sequential forward selection strategy to find a set of important features. 

Used Random Forest Regressor learning method for measuring the performance (accuracy) in the wrapper approach.

This used [stratified](https://www.scribbr.com/methodology/stratified-sampling/) 5-fold cross-validation for measuring accuracy. And the program will keep
adding the features as long as there is some improvement in the classification accuracy or 75%
features have been selected. The output of the program will be the set of important features on the console.

## Dataset

[messidor_features](https://github.com/TP1232/K_fold/blob/main/messidor_features.arff)

## Usage

```bash
python k_fold.py
```



