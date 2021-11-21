Environment:
Python 3.9.8
    with Scikit-learn version 1.0.1
    and Imbalanced-learn version 0.8.1
    and NLKT version 3.6.5
    and pandas version 1.3.4

Citations:
1. Bird, Steven, Edward Loper and Ewan Klein (2009), Natural Language Processing with Python. O’Reilly Media Inc.
2. Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.

----------------------------------------------------
Initial Running Result:

**1st time try:**

Average Accuracy of Decision Tree: 0.794545

Average Accuracy of Random Forest: 0.830909

Average Accuracy of Logistic Regression: 0.816364

Average Accuracy of Support Vector Machine: 0.830909

**2nd time try:**

Average Accuracy of Decision Tree: 0.805455

Average Accuracy of Random Forest: 0.829091

Average Accuracy of Logistic Regression: 0.816364

Average Accuracy of Support Vector Machine: 0.830909

----------------------------------------------------
SMOTE added: (We can see a decline here)

**1st time try:**

Average Accuracy of Decision Tree: 0.630909

Average Accuracy of Random Forest: 0.649091

Average Accuracy of Logistic Regression: 0.665455

Average Accuracy of Support Vector Machine: 0.830909

**2nd time try:**

Average Accuracy of Decision Tree: 0.612727

Average Accuracy of Random Forest: 0.680000

Average Accuracy of Logistic Regression: 0.663636

Average Accuracy of Support Vector Machine: 0.830909

-----------------------------------------------------
Modifying Hyperparameters: (All models perform equally well in this case)

Average Accuracy of Decision Tree: 0.830909

Average Accuracy of Random Forest: 0.830909

Average Accuracy of Logistic Regression: 0.830909

Average Accuracy of Support Vector Machine: 0.830909


Entire Script's Output:
Dataset Loaded.
#1 iteration of 10 fold cross val:
Examining Decision Tree
The Accuracy of Decision Tree with C (0.1) and M (2) is: 0.800000 , which is the best case
Examining Random Forest
The Accuracy of Random Forest with I (1000) and K (50) is: 0.800000 , which is the best case
Examining SVM
The Accuracy of SVM with gamma (857) and K (857) is: 0.800000 , which is the best case
Examining Logistic Regression
The Accuracy of Logistic Regression with ridge (900) is: 0.800000 , which is the best case
#2 iteration of 10 fold cross val:
Examining Decision Tree
The Accuracy of Decision Tree with C (0.1) and M (2) is: 0.981818 , which is the best case
Examining Random Forest
The Accuracy of Random Forest with I (1000) and K (50) is: 0.981818 , which is the best case
Examining SVM
The Accuracy of SVM with gamma (857) and K (857) is: 0.981818 , which is the best case
Examining Logistic Regression
The Accuracy of Logistic Regression with ridge (900) is: 0.981818 , which is the best case
#3 iteration of 10 fold cross val:
Examining Decision Tree
The Accuracy of Decision Tree with C (0.1) and M (2) is: 0.981818 , which is the best case
Examining Random Forest
The Accuracy of Random Forest with I (1000) and K (50) is: 0.981818 , which is the best case
Examining SVM
The Accuracy of SVM with gamma (857) and K (857) is: 0.981818 , which is the best case
Examining Logistic Regression
The Accuracy of Logistic Regression with ridge (900) is: 0.981818 , which is the best case
#4 iteration of 10 fold cross val:
Examining Decision Tree
The Accuracy of Decision Tree with C (0.1) and M (2) is: 0.818182 , which is the best case
Examining Random Forest
The Accuracy of Random Forest with I (1000) and K (50) is: 0.818182 , which is the best case
Examining SVM
The Accuracy of SVM with gamma (857) and K (857) is: 0.818182 , which is the best case
Examining Logistic Regression
The Accuracy of Logistic Regression with ridge (900) is: 0.818182 , which is the best case
#5 iteration of 10 fold cross val:
Examining Decision Tree
The Accuracy of Decision Tree with C (0.1) and M (2) is: 0.781818 , which is the best case
Examining Random Forest
The Accuracy of Random Forest with I (1000) and K (50) is: 0.763636 , which is the best case
Examining SVM
The Accuracy of SVM with gamma (857) and K (857) is: 0.781818 , which is the best case
Examining Logistic Regression
The Accuracy of Logistic Regression with ridge (900) is: 0.781818 , which is the best case
#6 iteration of 10 fold cross val:
Examining Decision Tree
The Accuracy of Decision Tree with C (0.1) and M (2) is: 0.690909 , which is the best case
Examining Random Forest
The Accuracy of Random Forest with I (1000) and K (50) is: 0.690909 , which is the best case
Examining SVM
The Accuracy of SVM with gamma (857) and K (857) is: 0.690909 , which is the best case
Examining Logistic Regression
The Accuracy of Logistic Regression with ridge (900) is: 0.690909 , which is the best case
#7 iteration of 10 fold cross val:
Examining Decision Tree
The Accuracy of Decision Tree with C (0.1) and M (2) is: 0.690909 , which is the best case
Examining Random Forest
The Accuracy of Random Forest with I (1000) and K (50) is: 0.690909 , which is the best case
Examining SVM
The Accuracy of SVM with gamma (857) and K (857) is: 0.690909 , which is the best case
Examining Logistic Regression
The Accuracy of Logistic Regression with ridge (900) is: 0.690909 , which is the best case
#8 iteration of 10 fold cross val:
Examining Decision Tree
The Accuracy of Decision Tree with C (0.1) and M (2) is: 0.672727 , which is the best case
Examining Random Forest
The Accuracy of Random Forest with I (1000) and K (50) is: 0.672727 , which is the best case
Examining SVM
The Accuracy of SVM with gamma (857) and K (857) is: 0.672727 , which is the best case
Examining Logistic Regression
The Accuracy of Logistic Regression with ridge (900) is: 0.672727 , which is the best case
#9 iteration of 10 fold cross val:
Examining Decision Tree
The Accuracy of Decision Tree with C (0.1) and M (2) is: 0.963636 , which is the best case
Examining Random Forest
The Accuracy of Random Forest with I (1000) and K (50) is: 0.963636 , which is the best case
Examining SVM
The Accuracy of SVM with gamma (857) and K (857) is: 0.963636 , which is the best case
Examining Logistic Regression
The Accuracy of Logistic Regression with ridge (900) is: 0.963636 , which is the best case
#10 iteration of 10 fold cross val:
Examining Decision Tree
The Accuracy of Decision Tree with C (0.1) and M (2) is: 0.927273 , which is the best case
Examining Random Forest
The Accuracy of Random Forest with I (1000) and K (50) is: 0.945455 , which is the best case
Examining SVM
The Accuracy of SVM with gamma (857) and K (857) is: 0.927273 , which is the best case
Examining Logistic Regression
The Accuracy of Logistic Regression with ridge (900) is: 0.927273 , which is the best case
Average Accuracy of Decision Tree: 0.830909
Average Accuracy of Random Forest: 0.830909
Average Accuracy of Logistic Regression: 0.830909
Average Accuracy of Support Vector Machine: 0.830909