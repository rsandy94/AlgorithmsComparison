import pandas
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, accuracy_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Dataset1
df = pandas.read_csv("energydata_complete.csv")
df = df.drop(['date'], axis=1)
df['Efficiency'] = np.where(df['Appliances'] >= 60, "Good", "Bad")
df = df.drop(['Appliances'], axis=1)
x = df.drop(['Efficiency'], axis=1)
y = df['Efficiency']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1)
std_Xtrain = preprocessing.scale(x_train)
std_Xtest = preprocessing.scale(x_test)
# Linear kernel
C = [0.01, 0.5, 1, 10]
scores = []
for i in C:
    clf = SVC(C=i, kernel='linear', random_state=1)
    cv = KFold(n_splits=5, random_state=1, shuffle=True)
    scr = cross_val_score(clf, std_Xtrain, y_train, cv=cv)
    print(round(scr.mean(), 5))
    scores.append(round(scr.mean(), 5))

plt.figure()

plt.plot(C, scores, color='red',
         lw=2)
plt.xticks([0.01, 1, 5, 10])
plt.xlabel("Penalty parameter C")
plt.ylabel("Accuracy")
plt.show()
svclassifier = SVC(kernel='linear', probability=True, random_state=1, C=0.5)
svclassifier.fit(std_Xtrain, y_train)
ypred = svclassifier.predict(std_Xtest)
print(confusion_matrix(y_test, ypred))
# print(classification_report(y_test,ypred))
print(round(accuracy_score(y_test, ypred), 5))

prob = svclassifier.predict_proba(std_Xtest)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, prob, pos_label="Good")
tpr
plt.figure(figsize=(6, 4))

plt.plot(fpr, tpr, linewidth=2)

plt.plot([0, 1], [0, 1], 'k--')

plt.rcParams['font.size'] = 12

plt.title('ROC curve for RainTomorrow classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.show()
ROC_AUC = roc_auc_score(y_test, prob)
round(ROC_AUC, 5)

# Polynomial
degree = [2, 3, 4, 5]
scores1 = []
for i in degree:
    clf1 = SVC(C=1, kernel='poly', degree=i, random_state=1, gamma='auto')
    cv1 = KFold(n_splits=5, random_state=1, shuffle=True)
    scr1 = cross_val_score(clf1, std_Xtrain, y_train, cv=cv1)
    scores1.append(round(scr1.mean(), 5))
plt.figure()

plt.plot(degree, scores1, color='red',
         lw=2)
plt.xticks([2, 3, 4, 5])
plt.xlabel("Degree of polynomial")
plt.ylabel("Accuracy")
plt.show()
svclassifier1 = SVC(kernel='poly', degree=3, probability=True)
svclassifier1.fit(std_Xtrain, y_train)
ypred1 = svclassifier1.predict(std_Xtest)
print(confusion_matrix(y_test, ypred1))
# print(classification_report(y_test,ypred))
print(round(accuracy_score(y_test, ypred1), 5))
prob1 = svclassifier1.predict_proba(std_Xtest)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, prob1, pos_label="Good")
tpr
plt.figure(figsize=(6, 4))

plt.plot(fpr, tpr, linewidth=2)

plt.plot([0, 1], [0, 1], 'k--')

plt.rcParams['font.size'] = 12

plt.title('ROC curve for RainTomorrow classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.show()
ROC_AUC = roc_auc_score(y_test, prob1)
round(ROC_AUC, 5)

# RBF
C = [0.01, 0.5, 1, 10]
scores2 = []
for i in C:
    clf2 = SVC(kernel='rbf', random_state=1, C=i)
    cv2 = KFold(n_splits=5, random_state=1, shuffle=True)
    scr2 = cross_val_score(clf2, std_Xtrain, y_train, cv=cv2)
    scores2.append(round(scr2.mean(), 5))
plt.figure()

plt.plot(C, scores2, color='red',
         lw=2)
plt.xticks([0.01, 1, 5, 10])
plt.xlabel("Penalty term - C")
plt.ylabel("Accuracy")
plt.show()
svclassifier2 = SVC(kernel='rbf', probability=True, C=10)
svclassifier2.fit(std_Xtrain, y_train)
ypred2 = svclassifier2.predict(std_Xtest)
print(confusion_matrix(y_test, ypred2))
# print(classification_report(y_test,ypred))
print(round(accuracy_score(y_test, ypred2), 2))
prob2 = svclassifier2.predict_proba(std_Xtest)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, prob2, pos_label="Good")
tpr
plt.figure(figsize=(6, 4))

plt.plot(fpr, tpr, linewidth=2)

plt.plot([0, 1], [0, 1], 'k--')

plt.rcParams['font.size'] = 12

plt.title('ROC curve for RainTomorrow classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.show()
ROC_AUC = roc_auc_score(y_test, prob2)
round(ROC_AUC, 5)
# Decision Tree
clf_entropy = DecisionTreeClassifier(
    criterion="entropy", random_state=10)
clf_entropy.fit(std_Xtrain, y_train)
y_pred = clf_entropy.predict(std_Xtest)
print(confusion_matrix(y_test, y_pred))
print(round(accuracy_score(y_test, y_pred) * 100, 5))
clf_cv = DecisionTreeClassifier(criterion="entropy", random_state=10)
cv = KFold(n_splits=10, random_state=10, shuffle=True)
scr = cross_val_score(clf_cv, std_Xtrain, y_train, cv=cv)
print(round(scr.mean(), 5))
depth = list(range(2, 27 + 1, 1))
dpth_scores = []
for i in depth:
    clf_cv = DecisionTreeClassifier(criterion="entropy", random_state=10, max_depth=i)
    cv = KFold(n_splits=5, random_state=10, shuffle=True)
    scr = cross_val_score(clf_cv, std_Xtrain, y_train, cv=cv)
    print(round(scr.mean(), 5))
    dpth_scores.append(round(scr.mean(), 5))
dpth_scores1 = []
for i in depth:
    clf_train = DecisionTreeClassifier(criterion="entropy", random_state=10, max_depth=i)
    clf_train.fit(std_Xtrain, y_train)
    y_pred1 = clf_train.predict(std_Xtrain)
    print(round(accuracy_score(y_train, y_pred1), 5))
    dpth_scores1.append(round(accuracy_score(y_train, y_pred1), 5))
plt.figure()
plt.plot(depth, dpth_scores, dpth_scores1)
plt.xticks([2, 5, 8, 10, 12, 15, 17, 20, 23, 25, 27])
plt.legend(["Cross validation accuracy", "Training Accuracy"])
plt.show()
# Depth=7
clf_entropy = DecisionTreeClassifier(
    criterion="entropy", random_state=10, max_depth=7)
clf_entropy.fit(std_Xtrain, y_train)
y_pred = clf_entropy.predict(std_Xtest)
print(confusion_matrix(y_test, y_pred))
print(round(accuracy_score(y_test, y_pred) * 100, 5))
classification_report(y_test, y_pred)

probs = clf_entropy.predict_proba(x_test)[:, 1]

fpr, tpr, th = roc_curve(y_test, probs, pos_label="Good")

plt.figure(figsize=(6, 4))

plt.plot(fpr, tpr, linewidth=2)

plt.plot([0, 1], [0, 1], 'k--')

plt.rcParams['font.size'] = 12

plt.title('ROC curve for RainTomorrow classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.show()
ROC_AUC = roc_auc_score(y_test, probs)
round(ROC_AUC, 5)
# Gradient boosting
n_est = [20, 50, 70, 100]
est_scores = []
for n in n_est:
    gb_clf = GradientBoostingClassifier(random_state=10, n_estimators=n)
    cv = KFold(n_splits=5, random_state=10, shuffle=True)
    scr = cross_val_score(gb_clf, std_Xtrain, y_train, cv=cv)
    est_scores.append(round(scr.mean(), 5))
plt.figure()
plt.plot(n_est, est_scores)
plt.xticks([20, 50, 70, 100])
plt.show()
alpha = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]
alpha_scores = []
for a in alpha:
    gb_clf = GradientBoostingClassifier(random_state=10, n_estimators=100, learning_rate=a)
    cv = KFold(n_splits=5, random_state=10, shuffle=True)
    scr = cross_val_score(gb_clf, std_Xtrain, y_train, cv=cv)
    print(round(scr.mean(), 5))
    alpha_scores.append(round(scr.mean(), 5))
plt.figure()
plt.plot(alpha, alpha_scores)
plt.xticks([0.1, 0.5, 0.75, 1])
plt.show()
gb_clf.score(std_Xtrain, y_train)
gb_clf.score(std_Xtest, y_test)

depth = list(range(2, 10 + 1, 1))
dpth_scores1 = []
for i in depth:
    gb_clf = GradientBoostingClassifier(random_state=10, n_estimators=100, learning_rate=0.75, max_depth=i)
    cv = KFold(n_splits=5, random_state=10, shuffle=True)
    scr = cross_val_score(gb_clf, std_Xtrain, y_train, cv=cv)
    print(round(scr.mean(), 5))
    dpth_scores1.append(round(scr.mean(), 5))
depth = list(range(2, 10 + 1, 1))
dpth_scores = []
for i in depth:
    gb_clf = GradientBoostingClassifier(random_state=10, n_estimators=100, learning_rate=0.75, max_depth=i)
    gb_clf.fit(std_Xtrain, y_train)
    print(round(gb_clf.score(std_Xtrain, y_train), 5))
    dpth_scores.append(round(gb_clf.score(std_Xtrain, y_train), 5))
import matplotlib.pyplot as plt

plt.figure()
plt.plot(depth, dpth_scores)
plt.plot(depth, dpth_scores1)
plt.xticks([2, 4, 6, 8, 10])
plt.legend(["Training Accuracy", "Cross validation accuracy"])
plt.show()
gb_clf = GradientBoostingClassifier(random_state=10, n_estimators=100, learning_rate=0.75, max_depth=3)
gb_clf.fit(std_Xtrain, y_train)
y_pred = gb_clf.predict(std_Xtest)
print(confusion_matrix(y_test, y_pred))
print(round(accuracy_score(y_test, y_pred) * 100, 5))
print(classification_report(y_test, y_pred))
probs = gb_clf.predict_proba(std_Xtest)[:, 1]

fpr, tpr, th = roc_curve(y_test, probs, pos_label="Good")

plt.figure(figsize=(6, 4))

plt.plot(fpr, tpr, linewidth=2)

plt.plot([0, 1], [0, 1], 'k--')

plt.rcParams['font.size'] = 12

plt.title('ROC curve for RainTomorrow classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.show()
ROC_AUC = roc_auc_score(y_test, probs)
round(ROC_AUC, 5)

#Second dataset
df1 = pandas.read_csv("weatherAUS.csv")
df1.describe(include="all")
df1['MinTemp'] = df1['MinTemp'].fillna((df1['MinTemp'].mean()))
df1['MaxTemp'] = df1['MaxTemp'].fillna((df1['MaxTemp'].mean()))
df1['Rainfall'] = df1['Rainfall'].fillna((df1['Rainfall'].mean()))
df1['Evaporation'] = df1['Evaporation'].fillna((df1['Evaporation'].mean()))
df1['Sunshine'] = df1['Sunshine'].fillna((df1['Sunshine'].mean()))
df1['WindGustSpeed'] = df1['WindGustSpeed'].fillna((df1['WindGustSpeed'].mean()))
df1['WindSpeed9am'] = df1['WindSpeed9am'].fillna((df1['WindSpeed9am'].mean()))
df1['WindSpeed3pm'] = df1['WindSpeed3pm'].fillna((df1['WindSpeed3pm'].mean()))
df1['Humidity9am'] = df1['Humidity9am'].fillna((df1['Humidity9am'].mean()))
df1['Humidity3pm'] = df1['Humidity3pm'].fillna((df1['Humidity3pm'].mean()))
df1['Pressure9am'] = df1['Pressure9am'].fillna((df1['Pressure9am'].mean()))
df1['Pressure3pm'] = df1['Pressure3pm'].fillna((df1['Pressure3pm'].mean()))
df1['Cloud9am'] = df1['Cloud9am'].fillna((df1['Cloud9am'].mean()))
df1['Cloud3pm'] = df1['Cloud3pm'].fillna((df1['Cloud3pm'].mean()))
df1['Temp3pm'] = df1['Temp3pm'].fillna((df1['Temp3pm'].mean()))
df1['Temp9am'] = df1['Temp9am'].fillna((df1['Temp9am'].mean()))
df1['RISK_MM'] = df1['RISK_MM'].fillna((df1['RISK_MM'].mean()))
df1['WindGustDir'] = df1['WindGustDir'].fillna("W")
df1['WindDir9am'] = df1['WindDir9am'].fillna("N")
df1['WindDir3pm'] = df1['WindDir3pm'].fillna("SE")
df1['RainToday'] = df1['RainToday'].fillna("No")
df1 = df1.drop(["Date", "Location", "RISK_MM"], axis=1)
df1 = df1.drop('WindGustDir', axis=1)
df1 = df1.drop('WindDir3pm', axis=1)
df1 = df1.drop('WindDir9am', axis=1)
df1 = df1.drop('RainToday', axis=1)
df1['RainTomorrow'].replace({'No': 0, 'Yes': 1}, inplace=True)

num_cols = df1.columns[df1.dtypes.apply(lambda c: np.issubdtype(c, np.number))]

z = np.abs(stats.zscore(df1[num_cols]))
print(z)
df1 = df1[(z < 3).all(axis=1)]
print(df1.shape)

df2 = df1.sample(n=20000, random_state=1)
x = df2.drop(['RainTomorrow'], axis=1)
y = df2['RainTomorrow']
x_train1, x_test1, y_train1, y_test1 = train_test_split(x, y, train_size=0.7, random_state=1, shuffle=True)
std_Xtrain1 = preprocessing.scale(x_train1)
std_Xtest1 = preprocessing.scale(x_test1)
scores = []
C = [0.01, 0.5, 1, 10]
for i in C:
    clf1 = SVC(kernel='linear', random_state=1, C=i)
    cv1 = KFold(n_splits=5, random_state=1, shuffle=True)
    scr1 = cross_val_score(clf1, std_Xtrain1, y_train1, cv=cv1)
    scores.append(round(scr1.mean(), 5))
plt.figure()

plt.plot(C, scores, color='red',
         lw=2)
plt.xticks([0.01, 1, 5, 10])
plt.xlabel("Penalty parameter C")
plt.ylabel("Accuracy")
plt.show()

# Linear kernel
svclassifier = SVC(kernel='linear', random_state=1, C=0.5, probability=True)
svclassifier.fit(std_Xtrain1, y_train1)
ypred = svclassifier.predict(std_Xtest1)
print(confusion_matrix(y_test1, ypred))
# print(classification_report(y_test,ypred))
print(round(accuracy_score(y_test1, ypred), 5))

prob = svclassifier.predict_proba(std_Xtest1)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test1, prob, pos_label=1)
tpr
plt.figure(figsize=(6, 4))

plt.plot(fpr, tpr, linewidth=2)

plt.plot([0, 1], [0, 1], 'k--')

plt.rcParams['font.size'] = 12

plt.title('ROC curve for RainTomorrow classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.show()

ROC_AUC = roc_auc_score(y_test1, prob)
round(ROC_AUC, 5)
#Polynomial
degree = [2, 3, 4, 5]
scores1 = []
for i in degree:
    clf1 = SVC(kernel='poly', degree=i, random_state=1, gamma="auto")
    cv1 = KFold(n_splits=5, random_state=1, shuffle=True)
    scr1 = cross_val_score(clf1, std_Xtrain1, y_train1, cv=cv1)
    print(scores1)
    scores1.append((round(scr1.mean(), 5)))
plt.figure()

plt.plot(degree, scores1, color='red',
         lw=2)
plt.xticks([2, 3, 4, 5])
plt.xlabel("Degree of polynomial")
plt.ylabel("Accuracy")
plt.show()
svclassifier1 = SVC(kernel='poly', degree=3, random_state=1, gamma="auto", probability=True)
svclassifier1.fit(std_Xtrain1, y_train1)
ypred1 = svclassifier1.predict(std_Xtest1)
print(confusion_matrix(y_test1, ypred1))
# print(classification_report(y_test,ypred))
print(round(accuracy_score(y_test1, ypred1), 5))

prob2 = svclassifier1.predict_proba(std_Xtest1)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test1, prob2, pos_label=1)

plt.figure(figsize=(6, 4))

plt.plot(fpr, tpr, linewidth=2)

plt.plot([0, 1], [0, 1], 'k--')

plt.rcParams['font.size'] = 12

plt.title('ROC curve for RainTomorrow classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.show()

ROC_AUC = roc_auc_score(y_test1, prob2)
round(ROC_AUC, 5)
#Rbf
C = [0.01, 0.5, 1, 10]
scores2 = []
for i in C:
    clf2 = SVC(kernel='rbf', random_state=1, C=i, gamma='auto')
    cv2 = KFold(n_splits=5, random_state=1, shuffle=True)
    scr2 = cross_val_score(clf2, std_Xtrain1, y_train1, cv=cv2)

    scores2.append(round(scr2.mean(), 5))
plt.figure()

plt.plot(C, scores2, color='red',
         lw=2)
plt.xticks([0.01, 1, 5, 10])
plt.xlabel("Penalty parameter C")
plt.ylabel("Accuracy")
plt.show()
svclassifier2 = SVC(kernel='rbf', random_state=1, gamma='auto', C=1, probability=True)
svclassifier2.fit(std_Xtrain1, y_train1)
ypred2 = svclassifier2.predict(std_Xtest1)
print(confusion_matrix(y_test1, ypred2))
# print(classification_report(y_test,ypred))
print(round(accuracy_score(y_test1, ypred2), 4))
prob3 = svclassifier2.predict_proba(std_Xtest1)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test1, prob3, pos_label=1)
tpr
plt.figure(figsize=(6, 4))

plt.plot(fpr, tpr, linewidth=2)

plt.plot([0, 1], [0, 1], 'k--')

plt.rcParams['font.size'] = 12

plt.title('ROC curve for RainTomorrow classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.show()
ROC_AUC = roc_auc_score(y_test1, prob3)
round(ROC_AUC, 5)
#Decision Tree
clf_entropy = DecisionTreeClassifier(
    criterion="entropy", random_state=10)
clf_entropy.fit(std_Xtrain1, y_train1)
y_pred = clf_entropy.predict(std_Xtest1)
print(confusion_matrix(y_test1, y_pred))
print(round(accuracy_score(y_test1, y_pred) * 100, 5))
clf_cv = DecisionTreeClassifier(criterion="entropy", random_state=10)
cv = KFold(n_splits=10, random_state=10, shuffle=True)
scr = cross_val_score(clf_cv, std_Xtrain1, y_train1, cv=cv)
print(round(scr.mean(), 5))
depth = list(range(2, 16 + 1, 1))
dpth_scores = []
for i in depth:
    clf_cv = DecisionTreeClassifier(criterion="entropy", random_state=10, max_depth=i)
    cv = KFold(n_splits=5, random_state=10, shuffle=True)
    scr = cross_val_score(clf_cv, std_Xtrain1, y_train1, cv=cv)
    print(round(scr.mean(), 5))
    dpth_scores.append(round(scr.mean(), 5))

for i in depth:
    clf_train = DecisionTreeClassifier(criterion="entropy", random_state=10, max_depth=i)
    clf_train.fit(std_Xtrain1, y_train1)
    y_pred1 = clf_train.predict(std_Xtrain1)
    print(round(accuracy_score(y_train1, y_pred1), 5))
    dpth_scores1.append(round(accuracy_score(y_train1, y_pred1), 5))
plt.figure()
plt.plot(depth, dpth_scores)
plt.plot(depth, dpth_scores1)

plt.xticks([2, 4, 6, 8, 10, 12, 14, 16, 18])
plt.legend(["Cross validation accuracy", "Training Accuracy"])
plt.show()
clf_entropy = DecisionTreeClassifier(
    criterion="entropy", random_state=10, max_depth=5)
clf_entropy.fit(std_Xtrain1, y_train1)
y_pred = clf_entropy.predict(std_Xtest1)
print(confusion_matrix(y_test1, y_pred))
print(round(accuracy_score(y_test1, y_pred) * 100, 5))
print(classification_report(y_test1, y_pred))
probs1 = clf_entropy.predict_proba(std_Xtest1)[:, 1]

fpr, tpr, th = roc_curve(y_test1, probs1, pos_label=1)
plt.figure(figsize=(6, 4))

plt.plot(fpr, tpr, linewidth=2)

plt.plot([0, 1], [0, 1], 'k--')

plt.rcParams['font.size'] = 12

plt.title('ROC curve for RainTomorrow classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.show()
ROC_AUC = roc_auc_score(y_test1, probs1)
round(ROC_AUC, 5)
#Gradient Boosting
n_est = [20, 50, 70, 100]
est_scores = []
for n in n_est:
    gb_clf = GradientBoostingClassifier(random_state=10, n_estimators=n)
    cv = KFold(n_splits=5, random_state=10, shuffle=True)
    scr = cross_val_score(gb_clf, std_Xtrain1, y_train1, cv=cv)
    est_scores.append(round(scr.mean(), 5))
import matplotlib.pyplot as plt

plt.figure()
plt.plot(n_est, est_scores)
plt.xticks([20, 50, 70, 100])

plt.show()
alpha = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]
alpha_scores = []
for a in alpha:
    gb_clf = GradientBoostingClassifier(random_state=10, n_estimators=50, learning_rate=a)
    cv = KFold(n_splits=5, random_state=10, shuffle=True)
    scr = cross_val_score(gb_clf, std_Xtrain1, y_train1, cv=cv)
    print(round(scr.mean(), 5))
    alpha_scores.append(round(scr.mean(), 5))
plt.figure()
plt.plot(alpha, alpha_scores)
plt.xticks([0.1, 0.5, 0.75, 1])
plt.show()
depth = list(range(2, 11 + 1, 1))
dpth_scores1 = []
for i in depth:
    gb_clf = GradientBoostingClassifier(random_state=10, n_estimators=50, learning_rate=0.075, max_depth=i)
    cv = KFold(n_splits=5, random_state=10, shuffle=True)
    scr = cross_val_score(gb_clf, std_Xtrain1, y_train1, cv=cv)
    print(round(scr.mean(), 5))
    dpth_scores1.append(round(scr.mean(), 5))
depth = list(range(2, 11 + 1, 1))
dpth_scores = []
for i in depth:
    gb_clf = GradientBoostingClassifier(random_state=10, n_estimators=50, learning_rate=0.075, max_depth=i)
    gb_clf.fit(std_Xtrain1, y_train1)
    print(round(gb_clf.score(std_Xtrain1, y_train1), 5))
    dpth_scores.append(round(gb_clf.score(std_Xtrain1, y_train1), 5))
plt.figure()
plt.plot(depth, dpth_scores)
plt.plot(depth, dpth_scores1)
plt.xticks([2, 4, 6, 8, 10])
plt.legend(["Training Accuracy", "Cross validation accuracy"])
plt.show()
gb_clf = GradientBoostingClassifier(random_state=10, n_estimators=50, learning_rate=0.075, max_depth=4)
gb_clf.fit(std_Xtrain1, y_train1)
y_pred = gb_clf.predict(std_Xtest1)
print(confusion_matrix(y_test1, y_pred))
print(round(accuracy_score(y_test1, y_pred) * 100, 5))
print(classification_report(y_test1, y_pred))
probs1 = gb_clf.predict_proba(std_Xtest1)[:, 1]

fpr, tpr, th = roc_curve(y_test1, probs1, pos_label=1)
plt.figure(figsize=(6, 4))

plt.plot(fpr, tpr, linewidth=2)

plt.plot([0, 1], [0, 1], 'k--')

plt.rcParams['font.size'] = 12

plt.title('ROC curve for RainTomorrow classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.show()
ROC_AUC = roc_auc_score(y_test1, probs1)
round(ROC_AUC, 5)

# Bar charts for first dataset

objects = ('SVM Linear', 'SVM Polynomial', 'SVM Radial', 'Decison Tree', 'Gradient Boosintg')
y_pos = np.arange(len(objects))
performance = [0.73788, 0.76642, 0.8, 0.74869, 0.79075]

plt.barh(y_pos, performance, align='center', alpha=0.5)
plt.yticks(y_pos, objects)
plt.xlabel('Accuracy')
plt.ylabel('Algorithms')
plt.title('Comparison of all models')
for i, v in enumerate(performance):
    plt.text(v + 0.001, i, str(v), color='blue', ha='right', va='center')
plt.show()

# Roc curves for first dataset
svclassifier = SVC(kernel='linear', probability=True, random_state=1, C=0.5)
svclassifier.fit(std_Xtrain, y_train)
prob1 = svclassifier.predict_proba(std_Xtest)[:, 1]

# Polynomial kernel

svclassifier1 = SVC(kernel='poly', degree=3, probability=True)
svclassifier1.fit(std_Xtrain, y_train)
prob2 = svclassifier1.predict_proba(std_Xtest)[:, 1]

# Sigmoid
svclassifier2 = SVC(kernel='rbf', probability=True, C=10)
svclassifier2.fit(std_Xtrain, y_train)
prob3 = svclassifier2.predict_proba(std_Xtest)[:, 1]

# Depth=7
clf_entropy = DecisionTreeClassifier(
    criterion="entropy", random_state=10, max_depth=7)
clf_entropy.fit(std_Xtrain, y_train)
prob4 = clf_entropy.predict_proba(std_Xtest)[:, 1]

gb_clf = GradientBoostingClassifier(random_state=10, n_estimators=100, learning_rate=0.75, max_depth=3)
gb_clf.fit(std_Xtrain, y_train)
prob5 = gb_clf.predict_proba(std_Xtest)[:, 1]

fpr1, tpr1, th1 = roc_curve(y_test, prob1, pos_label="Good")
fpr2, tpr2, th2 = roc_curve(y_test, prob2, pos_label="Good")
fpr3, tpr3, th3 = roc_curve(y_test, prob3, pos_label="Good")
fpr4, tpr4, th4 = roc_curve(y_test, prob4, pos_label="Good")
fpr5, tpr5, th5 = roc_curve(y_test, prob5, pos_label="Good")

# roc_auc = metrics.auc(fpr, tpr)
# fpr, tpr, = roc_curve(y_test, ypred,pos_label="Good", drop_intermediate=False)
plt.figure(figsize=(6, 4))

plt.plot(fpr1, tpr1, linewidth=2)
plt.plot(fpr2, tpr2, linewidth=2)
plt.plot(fpr3, tpr3, linewidth=2)
plt.plot(fpr4, tpr4, linewidth=2)
plt.plot(fpr5, tpr5, linewidth=2)

plt.plot([0, 1], [0, 1], 'k--')

plt.rcParams['font.size'] = 12

plt.title('ROC curve comparison for all Algorithms')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')
plt.legend(
    ["SVM-Linear_AUC=0.81251", "SVM-Polynomial_AUC=0.85136", "SVM-Radial_AUC=0.88014", "DecisonTree_AUC= 0.81959",
     "GradientBoosting_AUC= 0.86579"])

plt.show()

# No. of training set size vs Accuracy for dataset1
n = [5000, 10000, 15000, 19735]
scores_tr = []
scores_cv = []
for i in n:
    df1 = df.sample(n=i, random_state=1)
    x = df1.drop(['Efficiency'], axis=1)
    y = df1['Efficiency']
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1)
    std_Xtrain = preprocessing.scale(x_train)
    std_Xtest = preprocessing.scale(x_test)
    svclassifier = SVC(kernel='linear', probability=True, random_state=1, C=0.5)
    svclassifier.fit(std_Xtrain, y_train)
    ypred = svclassifier.predict(std_Xtest)
    print(round(accuracy_score(y_test, ypred), 5))
    scores_tr.append(round(accuracy_score(y_test, ypred), 5))

    clf = SVC(C=0.5, kernel='linear', random_state=1)
    cv = KFold(n_splits=5, random_state=1, shuffle=True)
    scr = cross_val_score(clf, std_Xtrain, y_train, cv=cv)
    print(round(scr.mean(), 5))
    scores_cv.append(round(scr.mean(), 5))

plt.figure()
plt.plot(n, scores_tr)
plt.plot(n, scores_cv)
plt.xticks(n)
plt.title("No. of samples vs Accuracy - SVM Linear ")
plt.xlabel("No. of samples")
plt.ylabel("Accuracy score")
plt.legend(["Train score", "CV score"])
plt.show()
scores_tr1 = []
scores_cv1 = []
for i in n:
    df1 = df.sample(n=i, random_state=1)
    x = df1.drop(['Efficiency'], axis=1)
    y = df1['Efficiency']
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1)
    std_Xtrain = preprocessing.scale(x_train)
    std_Xtest = preprocessing.scale(x_test)
    clf_entropy = DecisionTreeClassifier(
        criterion="entropy", random_state=10, max_depth=7)
    clf_entropy.fit(std_Xtrain, y_train)

    scores_tr1.append(round(clf_entropy.score(std_Xtest, y_test), 5))

    clf_cv = DecisionTreeClassifier(criterion="entropy", random_state=10, max_depth=7)
    cv = KFold(n_splits=5, random_state=10, shuffle=True)
    scr = cross_val_score(clf_cv, std_Xtrain, y_train, cv=cv)
    print(round(scr.mean(), 5))
    scores_cv1.append(round(scr.mean(), 5))
plt.figure()
plt.plot(n, scores_tr1)
plt.plot(n, scores_cv1)
plt.xticks(n)
plt.title("No. of samples vs Accuracy - Decision Tree(Depth=7) ")
plt.xlabel("No. of samples")
plt.ylabel("Accuracy score")
plt.legend(["Train score", "CV score"])
plt.show()
scores_tr2 = []
scores_cv2 = []
for i in n:
    df1 = df.sample(n=i, random_state=1)
    x = df1.drop(['Efficiency'], axis=1)
    y = df1['Efficiency']
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1)
    std_Xtrain = preprocessing.scale(x_train)
    std_Xtest = preprocessing.scale(x_test)
    gb_clf = GradientBoostingClassifier(random_state=10, n_estimators=100, learning_rate=0.75, max_depth=3)
    gb_clf.fit(std_Xtrain, y_train)
    print(gb_clf.score(std_Xtest, y_test))
    scores_tr2.append(round(gb_clf.score(std_Xtest, y_test), 5))

    gb_clf = GradientBoostingClassifier(random_state=10, n_estimators=100, learning_rate=0.75, max_depth=3)
    cv = KFold(n_splits=5, random_state=10, shuffle=True)
    scr = cross_val_score(gb_clf, std_Xtrain, y_train, cv=cv)
    print(round(scr.mean(), 5))
    scores_cv2.append(round(scr.mean(), 5))
plt.figure()
plt.plot(n, scores_tr2)
plt.plot(n, scores_cv2)
plt.xticks(n)
plt.title("No. of samples vs Accuracy - GradientBoosting(Depth=3) ")
plt.xlabel("No. of samples")
plt.ylabel("Accuracy score")
plt.legend(["Train score", "CV score"])
plt.show()

# Bar charts for second dataset
objects = ('SVM Linear', 'SVM Polynomial', 'SVM Radial', 'Decison Tree', 'Gradient Boosintg')
y_pos = np.arange(len(objects))
performance = [0.84383, 0.83883, 0.8478, 0.8323, 0.84816]

plt.barh(y_pos, performance, align='center', alpha=0.5)
plt.yticks(y_pos, objects)
plt.xlabel('Accuracy')
plt.ylabel('Algorithms')
plt.title('Comparison of all models')
for i, v in enumerate(performance):
    plt.text(v + 0.001, i, str(v), color='blue', ha='right', va='center')
plt.show()

# Roc curves for second dataset
svclassifier = SVC(kernel='linear', probability=True, random_state=1, C=0.5)
svclassifier.fit(std_Xtrain1, y_train1)
prob1 = svclassifier.predict_proba(std_Xtest1)[:, 1]

# Polynomial kernel

svclassifier1 = SVC(kernel='poly', degree=3, probability=True)
svclassifier1.fit(std_Xtrain1, y_train1)
prob2 = svclassifier1.predict_proba(std_Xtest1)[:, 1]

# Sigmoid
svclassifier2 = SVC(kernel='rbf', probability=True, C=10)
svclassifier2.fit(std_Xtrain1, y_train1)
prob3 = svclassifier2.predict_proba(std_Xtest1)[:, 1]

# Depth=7
clf_entropy = DecisionTreeClassifier(
    criterion="entropy", random_state=10, max_depth=7)
clf_entropy.fit(std_Xtrain1, y_train1)
prob4 = clf_entropy.predict_proba(std_Xtest1)[:, 1]

gb_clf = GradientBoostingClassifier(random_state=10, n_estimators=100, learning_rate=0.75, max_depth=3)
gb_clf.fit(std_Xtrain1, y_train1)
prob5 = gb_clf.predict_proba(std_Xtest1)[:, 1]
fpr1, tpr1, th1 = roc_curve(y_test1, prob1, pos_label=1)
fpr2, tpr2, th2 = roc_curve(y_test1, prob2, pos_label=1)
fpr3, tpr3, th3 = roc_curve(y_test1, prob3, pos_label=1)
fpr4, tpr4, th4 = roc_curve(y_test1, prob4, pos_label=1)
fpr5, tpr5, th5 = roc_curve(y_test1, prob5, pos_label=1)
plt.figure(figsize=(6, 4))
plt.plot(fpr1, tpr1, linewidth=2)
plt.plot(fpr2, tpr2, linewidth=2)
plt.plot(fpr3, tpr3, linewidth=2)
plt.plot(fpr4, tpr4, linewidth=2)
plt.plot(fpr5, tpr5, linewidth=2)

plt.plot([0, 1], [0, 1], 'k--')

plt.rcParams['font.size'] = 12

plt.title('ROC curve comparison for all Algorithms')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')
plt.legend(
    ["SVM-Linear_AUC=0.85673", "SVM-Polynomial_AUC=0.84083", "SVM-Radial_AUC= 0.82043", "DecisonTree_AUC= 0.81558",
     "GradientBoosting_AUC= 0.85725"])

plt.show()

# No. of training size Vs accuracy plots
n = [5000, 10000, 15000, 20000]
scores_tr = []
scores_cv = []
for i in n:
    df2 = df1.sample(n=i, random_state=1)
    x = df2.drop(['RainTomorrow'], axis=1)

    y = df2['RainTomorrow']
    x_train1, x_test1, y_train1, y_test1 = train_test_split(x, y, train_size=0.7, random_state=1, shuffle=True)
    # scaler=preprocessing.MinMaxScaler()
    # scaler.fit(x)
    std_Xtrain1 = preprocessing.scale(x_train1)
    std_Xtest1 = preprocessing.scale(x_test1)
    svclassifier = SVC(kernel='poly', degree=3, random_state=1, gamma="auto")
    svclassifier.fit(std_Xtrain1, y_train1)

    print(round(svclassifier.score(std_Xtest1, y_test1), 5))
    scores_tr.append(round(svclassifier.score(std_Xtest1, y_test1), 5))

    clf = SVC(kernel='poly', degree=3, random_state=1, gamma="auto")
    cv = KFold(n_splits=5, random_state=1, shuffle=True)
    scr = cross_val_score(clf, std_Xtrain1, y_train1, cv=cv)
    print(round(scr.mean(), 5))
    scores_cv.append(round(scr.mean(), 5))


plt.figure()
plt.plot(n, scores_tr)
plt.plot(n, scores_cv)
plt.xticks(n)
plt.title("No. of samples vs Accuracy - SVM Polynomial ")
plt.xlabel("No. of samples")
plt.ylabel("Accuracy score")
plt.legend(["Train score", "CV score"])
plt.show()

scores_tr1 = []
scores_cv1 = []
for i in n:
    df2 = df1.sample(n=i, random_state=1)
    x = df2.drop(['RainTomorrow'], axis=1)

    y = df2['RainTomorrow']
    x_train1, x_test1, y_train1, y_test1 = train_test_split(x, y, train_size=0.7, random_state=1, shuffle=True)
    # scaler=preprocessing.MinMaxScaler()
    # scaler.fit(x)
    std_Xtrain1 = preprocessing.scale(x_train1)
    std_Xtest1 = preprocessing.scale(x_test1)
    clf_entropy = DecisionTreeClassifier(
        criterion="entropy", random_state=10, max_depth=5)
    clf_entropy.fit(std_Xtrain1, y_train1)

    scores_tr1.append(round(clf_entropy.score(std_Xtest1, y_test1), 5))

    clf_cv = DecisionTreeClassifier(criterion="entropy", random_state=10, max_depth=5)
    cv = KFold(n_splits=5, random_state=10, shuffle=True)
    scr = cross_val_score(clf_cv, std_Xtrain1, y_train1, cv=cv)
    print(round(scr.mean(), 5))
    scores_cv1.append(round(scr.mean(), 5))

plt.figure()
plt.plot(n, scores_tr1)
plt.plot(n, scores_cv1)
plt.xticks(n)
plt.title("No. of samples vs Accuracy - Decision Tree(Depth=5) ")
plt.xlabel("No. of samples")
plt.ylabel("Accuracy score")
plt.legend(["Train score", "CV score"])
plt.show()

scores_tr2 = []
scores_cv2 = []
for i in n:
    df2 = df1.sample(n=i, random_state=1)
    x = df2.drop(['RainTomorrow'], axis=1)

    y = df2['RainTomorrow']
    x_train1, x_test1, y_train1, y_test1 = train_test_split(x, y, train_size=0.7, random_state=1, shuffle=True)
    std_Xtrain1 = preprocessing.scale(x_train1)
    std_Xtest1 = preprocessing.scale(x_test1)
    gb_clf = GradientBoostingClassifier(random_state=10, n_estimators=50, learning_rate=0.075, max_depth=4)
    gb_clf.fit(std_Xtrain1, y_train1)
    print(gb_clf.score(std_Xtest1, y_test1))
    scores_tr2.append(round(gb_clf.score(std_Xtest1, y_test1), 5))

    gb_clf1 = GradientBoostingClassifier(random_state=10, n_estimators=50, learning_rate=0.075, max_depth=4)
    cv = KFold(n_splits=5, random_state=10, shuffle=True)
    scr = cross_val_score(gb_clf1, std_Xtrain1, y_train1, cv=cv)
    print(round(scr.mean(), 5))
    scores_cv2.append(round(scr.mean(), 5))

plt.figure()
plt.plot(n, scores_tr2)
plt.plot(n, scores_cv2)
plt.xticks(n)
plt.title("No. of samples vs Accuracy - GradientBoosting(Depth=4) ")
plt.xlabel("No. of samples")
plt.ylabel("Accuracy score")
plt.legend(["Train score", "CV score"])
plt.show()
