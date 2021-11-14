import csv
import numpy
import statistics
from sklearn import tree
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from nltk.classify.scikitlearn import SklearnClassifier

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
        print("Imbalanced inputs, please check if you have sent the original labels and predicted labels to this function!")
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

### Classifications
testing_set_length = int(len(features)/10)
for x in range(0,10):
    print("#%d iteration of 10 fold cross val:" %(x+1))
    clf_dt = SklearnClassifier(tree.DecisionTreeClassifier()) # Decision Tree
    clf_rf = SklearnClassifier(RandomForestClassifier())      # Random Forest
    clf_lr = SklearnClassifier(LogisticRegression())          # Logistic Regression
    clf_svm = SklearnClassifier(make_pipeline(StandardScaler(with_mean=False), SVC(gamma='auto'))) # Support Vector Machine
    training_set = []
    testing_features = features[x * testing_set_length : x * testing_set_length + testing_set_length]
    testing_labels = labels[x * testing_set_length : x * testing_set_length + testing_set_length]
    for y in range(0, x*testing_set_length):
        training_set.append((features[y], labels[y]))
    for y in range(x * testing_set_length + testing_set_length, len(features)):
        training_set.append((features[y], labels[y]))
    clf_dt = clf_dt.train(training_set)
    clf_rf = clf_rf.train(training_set)
    clf_lr = clf_lr.train(training_set)
    clf_svm = clf_svm.train(training_set)
    res_of_dt.append(cal_acc(clf_dt.classify_many(testing_features), testing_labels))
    print("Accuracy of Decision Tree: %f" %(cal_acc(clf_dt.classify_many(testing_features), testing_labels)))
    res_of_rf.append(cal_acc(clf_rf.classify_many(testing_features), testing_labels))
    print("Accuracy of Random Forest: %f" %(cal_acc(clf_rf.classify_many(testing_features), testing_labels)))
    res_of_lr.append(cal_acc(clf_lr.classify_many(testing_features), testing_labels))
    print("Accuracy of Random Forest: %f" %(cal_acc(clf_lr.classify_many(testing_features), testing_labels)))
    res_of_svm.append(cal_acc(clf_svm.classify_many(testing_features), testing_labels))
    print("Accuracy of Random Forest: %f" %(cal_acc(clf_svm.classify_many(testing_features), testing_labels)))

print("Average Accuracy of Decision Tree: %f" %(statistics.mean(res_of_dt)))
print("Average Accuracy of Random Forest: %f" %(statistics.mean(res_of_rf)))
print("Average Accuracy of Logistic Regression: %f" %(statistics.mean(res_of_lr)))
print("Average Accuracy of Support Vector Machine: %f" %(statistics.mean(res_of_svm)))


'''
Citations:
1. Bird, Steven, Edward Loper and Ewan Klein (2009), Natural Language Processing with Python. O’Reilly Media Inc.
2. Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
'''