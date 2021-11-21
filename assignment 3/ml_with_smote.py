import csv
import warnings
import statistics
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from imblearn.over_sampling import SMOTE
from nltk.classify.scikitlearn import SklearnClassifier

'''
This python script applies SMOTE algorithm, which generates artificial instances in dataset to overcome imbalanced data, to optimize classification models
The testing sets remains original through the scripting
'''
__author__ = "Heng Zhang"
__email__ = "hzhan274@uOttawa.ca"

warnings.filterwarnings("ignore")

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

### Define a method which takes an array and outputs an dictionary
def dict_of(feature_list):
    dic = dict()
    for elem in feature_list:
        dic[elem] = elem
    return dic

### Define a method to convert str into int
def encode_str(string):
    if string == "":
        return 0
    int_list = list(string.encode('utf-8'))
    res = ""
    for elem in int_list:
        res = res + (str(elem) if elem >= 100 else "0"+str(elem))
    return int(res)

### Define a method to convert int into string using utf8
def decode_str(nb):
    nb = int(nb)
    if nb == 0:
        return ""
    str_of_nb = str(nb)
    if len(str_of_nb)%3 == 2:
        str_of_nb = "0"+str_of_nb
    res = ""
    for x in range(0,  int(len(str_of_nb)/3)):
        letter = str_of_nb[x*3:x*3+3]
        letter = chr(int(letter))
        res = res + letter
    return res

print("Dataset Loading.")
vectorizer = CountVectorizer()
### Extracting Dataset
extracted_features = []
extracted_labels = []
for sentence in csv_content:
    extracted_feature = [] ### a list contains all features for 1 sentence instance
    for x in range (2,22):
        extracted_feature.append(encode_str(sentence[x]))
    extracted_features.append(extracted_feature)
    extracted_labels.append(sentence[1])
### Data encoded to numeric

res_of_dt = []
res_of_rf = []
res_of_lr = []
res_of_svm = []


### Classifications
testing_set_length = int(len(extracted_features)/10)
for x in range(0,10):
    print("#%d iteration of 10 fold cross val:" %(x+1))
    clf_dt = SklearnClassifier(tree.DecisionTreeClassifier()) # Decision Tree
    clf_rf = SklearnClassifier(RandomForestClassifier())      # Random Forest
    clf_lr = SklearnClassifier(LogisticRegression())          # Logistic Regression
    clf_svm = SklearnClassifier(make_pipeline(StandardScaler(with_mean=False), SVC(gamma='auto'))) # Support Vector Machine
    training_features = []
    training_labels = []
    testing_features = extracted_features[x * testing_set_length : x * testing_set_length + testing_set_length]
    testing_labels = extracted_labels[x * testing_set_length : x * testing_set_length + testing_set_length]
    for y in range(0, x*testing_set_length):
        training_features.append(extracted_features[y])
        training_labels.append(extracted_labels[y])
    for y in range(x * testing_set_length + testing_set_length, len(extracted_features)):
        training_features.append(extracted_features[y])
        training_labels.append(extracted_labels[y])

    ### Scatter Diagram 1
    ### Distribution of feature 7 Over labels
    f7 =[]
    label = []
    for elem in training_labels:
        label.append(elem)
    for elem in training_features:
        f7.append(decode_str(elem[6]))
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.scatter(label, f7, color = "red", s = 3)
    ax1.set_yticklabels([])
    ax1.set_title("feature 7 (y) & label (x) before SMOTE")
    fig.set_figheight(15)
    fig.set_figwidth(10)

    # ### Distribution of feature 7 over feature 1
    # f7 =[]
    # f1 = []
    # for elem in training_features:
    #     f1.append(decode_str(elem[0]))
    #     f7.append(decode_str(elem[6]))
    # fig, ax = plt.subplots(2)
    # ax[0].scatter(f1, f7, color = "red")
    # ax[0].set_yticklabels([])
    # ax[0].set_title("feature 7 (y) & feature 1 (x) before SMOTE")
    # fig.set_figheight(15)
    # fig.set_figwidth(10)
    



    ### Apply SMOTE on training set
    training_features, training_labels = SMOTE(sampling_strategy = "minority").fit_resample(training_features, training_labels)
    print("SMOTE is done.")

    ### Scatter Diagram 2
    ### Distribution of feature 7 over feature 1
    f7 =[]
    label = []
    for elem in training_labels:
        label.append(elem)
    for elem in training_features:
        f7.append(decode_str(elem[6]))
    ax2.scatter(label, f7, color = "blue", s = 3)
    ax2.set_yticklabels([])
    ax2.set_title("feature 7 (y) & label (x) after SMOTE")

    # ### Distribution of feature 7 over feature 1
    # f7 =[]
    # f1 = []
    # for elem in training_features:
    #     f1.append(decode_str(elem[0]))
    #     f7.append(decode_str(elem[6]))
    # ax[1].scatter(f1, f7, color = "blue")
    # ax[1].set_yticklabels([])
    # ax[1].set_title("feature 7 (y) & feature 1 (x) after SMOTE")



    trf = [] ### Training Features
    counter = 0
    for elem in training_features:
        trf.append([])
        for elem2 in elem:
            trf[counter].append(decode_str(elem2))
        counter = counter + 1
    tef = [] ### Testing Features
    counter = 0
    for elem in testing_features:
        tef.append([])
        for elem2 in elem:
            tef[counter].append(decode_str(elem2))
        counter = counter + 1
    ### Data decoded.


    ### NLTK requires the data structure of a feature set to be dictionary
    ### Create lists of dictionaries
    training_features_d = []
    for feature_list in trf:
        training_features_d.append(dict_of(feature_list))
    testing_features_d = []
    for feature_list in tef:
        testing_features_d.append(dict_of(feature_list)) 
    
    training_set = []
    for x in range (0, len(training_features)):
        training_set.append([training_features_d[x], training_labels[x]])
    clf_dt = clf_dt.train(training_set)
    clf_rf = clf_rf.train(training_set)
    clf_lr = clf_lr.train(training_set)
    clf_svm = clf_svm.train(training_set)

    # Diagram Comments
    ax2.text(0.3,0.55,("Accuracy of Decision Tree (after SMOTE): %f\nAccuracy of Random Forest (after SMOTE): %f\n"
        "Accuracy of Logistic Regression (after SMOTE): %f\n"
        "Accuracy of Support Vector Machine (after SMOTE): %f" 
        %(cal_acc(clf_dt.classify_many(testing_features_d), testing_labels), 
        cal_acc(clf_rf.classify_many(testing_features_d), testing_labels), 
        cal_acc(clf_lr.classify_many(testing_features_d), testing_labels), 
        cal_acc(clf_svm.classify_many(testing_features_d), testing_labels))))
    plt.show()

    res_of_dt.append(cal_acc(clf_dt.classify_many(testing_features_d), testing_labels))
    print("Accuracy of Decision Tree: %f" %(cal_acc(clf_dt.classify_many(testing_features_d), testing_labels)))
    res_of_rf.append(cal_acc(clf_rf.classify_many(testing_features_d), testing_labels))
    print("Accuracy of Random Forest: %f" %(cal_acc(clf_rf.classify_many(testing_features_d), testing_labels)))
    res_of_lr.append(cal_acc(clf_lr.classify_many(testing_features_d), testing_labels))
    print("Accuracy of Logistic Regression: %f" %(cal_acc(clf_lr.classify_many(testing_features_d), testing_labels)))
    res_of_svm.append(cal_acc(clf_svm.classify_many(testing_features_d), testing_labels))
    print("Accuracy of Support Vector Machine: %f" %(cal_acc(clf_svm.classify_many(testing_features_d), testing_labels)))

print("Average Accuracy of Decision Tree: %f" %(statistics.mean(res_of_dt)))
print("Average Accuracy of Random Forest: %f" %(statistics.mean(res_of_rf)))
print("Average Accuracy of Logistic Regression: %f" %(statistics.mean(res_of_lr)))
print("Average Accuracy of Support Vector Machine: %f" %(statistics.mean(res_of_svm)))


'''
Citations:
1. Bird, Steven, Edward Loper and Ewan Klein (2009), Natural Language Processing with Python. O’Reilly Media Inc.
2. Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
'''