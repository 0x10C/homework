from utlis import prepare_data
from manual_dt import build_tree,five_cv_score
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

""" manual decison tree and five-fold cross validation """
raw_data,feature_names = prepare_data()
# print(raw_data.shape)
five_cv_score(raw_data,build_tree)



"""sklearn decision tree and five-fold cross validation"""

cancer = load_breast_cancer()
cancer_data = cancer['data']
cancer_target = cancer['target']
dt = DecisionTreeClassifier()
scores = cross_val_score(dt,cancer_data,cancer_target,cv=5,scoring='accuracy')
scores.mean()

