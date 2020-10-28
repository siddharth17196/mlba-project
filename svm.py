import pickle
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score

with open("data.pkl", "rb") as fil:
    data = pickle.load(fil)

data = shuffle(data, random_state=0)
x = [dat[:-1] for dat in data]
y = [dat[-1:] for dat in data]
clf = svm.SVC()
clf.fit(x, y)
# a = clf.predict(x)

scoring = ['precision_macro', 'recall_macro', 'f1_macro', 'accuracy']
scores = cross_validate(clf, x, y, cv=5, scoring=scoring)

for i in range(5):
    print(f"CV-{i+1}")
    print(f"Precision - {scores['test_precision_macro'][i]}")
    print(f"Recall - {scores['test_recall_macro'][i]}")
    print(f"F1_score - {scores['test_f1_macro'][i]}")
    print(f"Accuracy - {scores['test_accuracy'][i]}")
# print(accuracy_score(y, a))
# print(precision(y, a))
# print(recall(y, a))
# print(roc_auc_score(y, a))