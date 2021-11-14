import csv
import statistics
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
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
    features.append( dict( a = sentence[7], b = sentence[8], c = sentence[12], d = sentence[13], 
    e = sentence[14], f = sentence[17], g = sentence[19]))
    labels.append(sentence[1])
print("Dataset Loaded.")

res_of_dt = []
res_of_rf = []

testing_set_length = int(len(features)/10)
for x in range(0,10):
    print("#%d iteration of 10 fold cross val:" %(x+1))
    clf_dt = SklearnClassifier(tree.DecisionTreeClassifier())
    clf_rf = SklearnClassifier(RandomForestClassifier())
    training_set = []
    testing_features = features[x * testing_set_length : x * testing_set_length + testing_set_length]
    testing_labels = labels[x * testing_set_length : x * testing_set_length + testing_set_length]
    for y in range(0, x*testing_set_length):
        training_set.append((features[y], labels[y]))
    for y in range(x * testing_set_length + testing_set_length, len(features)):
        training_set.append((features[y], labels[y]))
    clf_dt = clf_dt.train(training_set)
    clf_rf = clf_rf.train(training_set)
    res_of_dt.append(cal_acc(clf_dt.classify_many(testing_features), testing_labels))
    print("Accuracy of Decision Tree: %f" %(cal_acc(clf_dt.classify_many(testing_features), testing_labels)))
    res_of_rf.append(cal_acc(clf_rf.classify_many(testing_features), testing_labels))
    print("Accuracy of Random Forest: %f" %(cal_acc(clf_rf.classify_many(testing_features), testing_labels)))

print("Average Accuracy of Decision Tree: %f" %(statistics.mean(res_of_dt)))
print("Average Accuracy of Random Forest: %f" %(statistics.mean(res_of_rf)))