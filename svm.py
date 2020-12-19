import pickle
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef, recall_score, precision_score, auc, roc_curve

from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score

with open("./features/nc_10_data.pkl", "rb") as fil:
    data = pickle.load(fil)

def metrics(y, preds, pred=None):
	print("Sensitivity-", recall_score(y, preds))
	print("Specificity-", precision_score(y, preds))
	print("Accuracy-", accuracy_score(y, preds))
	# fpr, tpr, thresholds = roc_curve(y, pred)
	# print("AUC-", auc(fpr, tpr))
	print("MCC-", matthews_corrcoef(y, preds))

data = shuffle(data, random_state=0)
x = [dat[:-1] for dat in data]
y = [dat[-1:][0] for dat in data]
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.15, random_state=42)
# parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
# svc = svm.SVC()
dt = DecisionTreeClassifier(random_state=0)
rf=RandomForestClassifier(random_state=0)
# parameters = {'criterion':['gini','entropy'],'max_depth':[10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]}
parameters = {'max_depth':[10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]}
clf = GridSearchCV(rf, parameters)

clf.fit(X_train, y_train)
preds = clf.predict(X_test)

metrics(y_test, preds)#, pred)

# scoring = ['precision_macro', 'recall_macro', 'f1_macro', 'accuracy']
# scores = cross_validate(clf, x, y, cv=5, scoring=scoring)

# for i in range(5):
    # print(f"CV-{i+1}")
    # print(f"Precision - {scores['test_precision_macro'][i]}")
    # print(f"Recall - {scores['test_recall_macro'][i]}")
    # print(f"F1_score - {scores['test_f1_macro'][i]}")
    # print(f"Accuracy - {scores['test_accuracy'][i]}")
# print(accuracy_score(y, a))
# print(precision(y, a))
# print(recall(y, a))
# print(roc_auc_score(y, a))