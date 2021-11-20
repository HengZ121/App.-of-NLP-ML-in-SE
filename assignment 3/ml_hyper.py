import csv
import numpy
import statistics
from sklearn import tree
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from nltk.classify.scikitlearn import SklearnClassifier

__author__ = "Heng Zhang"
__email__ = "hzhan274@uOttawa.ca"

### Read data from csv
csv_content = []
with open ("result.csv", 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        csv_content.append(row)

### Define a method to calculate accuracy of classifier
def cal_acc(list1, list2):
    counter = 0
    if len(list1) != len(list2):
        print("Insufficient labels/classification results, please check two arrays inputted!")
    for x in range(0,len(list1)):
        if list1[x] == list2[x]:
            counter = counter +1
    return counter/len(list1)

### Extracting Dataset
features = []
labels = []
for sentence in csv_content:
    features.append( dict( a = sentence[2], b = sentence[3], c = sentence[4], d = sentence[5], 
    e = sentence[6], f = sentence[7], g = sentence[8], h =  sentence[9], i = sentence[10], j = sentence[11],
    k = sentence[12], l = sentence[13], m = sentence[14], n = sentence[15], o = sentence[16], p = sentence[17],
    q = sentence[18], r = sentence[19], s = sentence[20], t = sentence[21]) )
    labels.append(sentence[1])
print("Dataset Loaded.")

res_of_dt = []
res_of_rf = []
res_of_lr = []
res_of_svm = []

'''
 Define a function to find Decision Tree model with best C and M
 C means the Cost-Complexity Pruning [0.1, 0.5] 5 steps
 M means the minimum number of instances per leaf [2, 10] 9 steps
 Return: The best accuracy value 
'''
def decision_tree(training_set, testing_features, testing_labels):
    print("Examining Decision Tree")
    best_accuracy = 0
    best_c = 0
    best_m = 0
    for c in range (1, 6):
        for m in range (2,11):
            clf_dt = SklearnClassifier(tree.DecisionTreeClassifier(ccp_alpha = (c/10), min_samples_leaf = m))
            clf_dt = clf_dt.train(training_set)
            accuracy = cal_acc(clf_dt.classify_many(testing_features), testing_labels)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_c = c
                best_m = m
    print("The Accuracy of Decision Tree with C (0.%d) and M (%d) is: %f , which is the best case" %(best_c, best_m, best_accuracy))
    return best_accuracy

'''
 Define a function to find Decision Tree model with best I and K
 I: # of trees    [100, 1000] 10 steps 
 K: # of features [5, 50] 10 steps
 Return: The best accuracy value 
'''
def random_forest(training_set, testing_features, testing_labels):
    print("Examining Random Forest")
    best_accuracy = 0
    best_i = 0
    best_k = 0
    for i in range (1, 11):
        for k in range (1, 11):
            clf_rf = SklearnClassifier(RandomForestClassifier(n_estimators = i*100, max_features = k*5, n_jobs = -1))
            clf_rf = clf_rf.train(training_set)
            accuracy = cal_acc(clf_rf.classify_many(testing_features), testing_labels)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_i = i
                best_k = k
    print("The Accuracy of Random Forest with I (%d) and K (%d) is: %f , which is the best case" %(i*100, k*5, best_accuracy))
    return best_accuracy

'''
 Define a function to find Logistic Regression (Ridge Regression) model with best ridge value
ridge    [1.0E-10, 1.0E3] = [0.0000000001, 1000] 10 steps
 Return: The best accuracy value 
'''
def ridge_regression(training_set, testing_features, testing_labels):
    print("Examining Logistic Regression")
    best_accuracy = 0
    best_ridge = 0
    for x in range (0, 10):
        ridge = x*(1.0E3 - 1e-10)/10 + 1e-10
        clf_rr = SklearnClassifier(RidgeClassifier(alpha = ridge))
        clf_rr = clf_rr.train(training_set)
        accuracy = cal_acc(clf_rr.classify_many(testing_features), testing_labels)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_ridge = ridge
    print("The Accuracy of Logistic Regression with ridge (%d) is: %f , which is the best case" %(ridge, best_accuracy))
    return best_accuracy

'''
 Define a function to find SVM model with best coef0 and gamma
 coef and gamma    [1.0E-3, 1.0E3] = [0.001, 1000] 7 steps
 Return: The best accuracy value 
'''
def svm(training_set, testing_features, testing_labels):
    print("Examining SVM")
    best_accuracy = 0
    best_g = 0
    best_c = 0
    for x in range (0, 7):
        for y in range (0, 7):
            clf_svm = SklearnClassifier(make_pipeline(StandardScaler(with_mean=False), SVC(gamma= x*999.999/7 + 0.001, coef0 = y*999.999/7 + 0.001)))
            clf_svm = clf_svm.train(training_set)
            accuracy = cal_acc(clf_svm.classify_many(testing_features), testing_labels)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_g = x
                best_c = y
    print("The Accuracy of SVM with gamma (%d) and K (%d) is: %f , which is the best case" %(x*999.999/7 + 0.001, y*999.999/7 + 0.001, best_accuracy))
    return best_accuracy


### Classifications
testing_set_length = int(len(features)/10)
for x in range(0,10):
    print("#%d iteration of 10 fold cross val:" %(x+1))

    training_set = []
    testing_features = features[x * testing_set_length : x * testing_set_length + testing_set_length]
    testing_labels = labels[x * testing_set_length : x * testing_set_length + testing_set_length]
    for y in range(0, x*testing_set_length):
        training_set.append((features[y], labels[y]))
    for y in range(x * testing_set_length + testing_set_length, len(features)):
        training_set.append((features[y], labels[y]))

    res_of_dt.append(decision_tree(training_set, testing_features, testing_labels))
    res_of_rf.append(random_forest(training_set, testing_features, testing_labels))
    res_of_svm.append(svm(training_set, testing_features, testing_labels))
    res_of_lr.append(ridge_regression(training_set, testing_features, testing_labels))

print("Average Accuracy of Decision Tree: %f" %(statistics.mean(res_of_dt)))
print("Average Accuracy of Random Forest: %f" %(statistics.mean(res_of_rf)))
print("Average Accuracy of Logistic Regression: %f" %(statistics.mean(res_of_lr)))
print("Average Accuracy of Support Vector Machine: %f" %(statistics.mean(res_of_svm)))


'''
Citations:
1. Bird, Steven, Edward Loper and Ewan Klein (2009), Natural Language Processing with Python. O’Reilly Media Inc.
2. Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
'''