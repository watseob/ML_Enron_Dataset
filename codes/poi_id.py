#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import numpy as np

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'bonus', 'total_payments', 'long_term_incentive', 'deferred_income',
                      'total_stock_value', 'restricted_stock', 'exercised_stock_options', 'expenses', 'other',
                'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    print data_dict


### Task 2: Remove outliers

data_dict.pop('TOTAL')
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')



### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

for name in data_dict:

    data_point = data_dict[name]

    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]

    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]

    if from_poi_to_this_person == 'NaN' or to_messages == 'NaN' :
        data_point["fraction_from_poi"] = "NaN"
    else :
        fraction_from_poi = float(from_poi_to_this_person) / float(to_messages)
        data_point["fraction_from_poi"] = fraction_from_poi

    if from_this_person_to_poi == 'NaN' or from_messages == 'NaN' :
        data_point['fraction_to_poi'] = "NaN"
    else :
        fraction_to_poi = float(from_this_person_to_poi) / float(from_messages)
        data_point["fraction_to_poi"] = fraction_to_poi


features_list += ['fraction_from_poi','fraction_to_poi']
remove_list = ['to_messages','from_messages','from_poi_to_this_person','from_this_person_to_poi']

for x in remove_list :
    features_list.remove(x)


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)



### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

clf_list = ['GaussianNB','LogisticRegression','SVM','KNN','AdaBoost']
tree = DecisionTreeClassifier(criterion='entropy', max_depth=None, random_state=0)

clf1 = GaussianNB()
clf2 = LogisticRegression(penalty='l2', C=0.001)
clf3 = SVC(kernel='rbf',C=10000)
clf4 = KNeighborsClassifier(n_neighbors=3)
clf5 = AdaBoostClassifier(base_estimator=tree,
                         n_estimators=1000,
                         learning_rate=0.01,
                         random_state=0)
pipe2 = Pipeline([['sc',StandardScaler()],
                  ['clf',clf2]])

pipe3 = Pipeline([['sc',StandardScaler()],
                  ['clf',clf5]])




### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.learning_curve import learning_curve
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import recall_score, f1_score

print "Variety of classifier"
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

for clf, label in zip([clf1,pipe2,clf3,clf4,clf5],clf_list) :
    clf.fit(features_train,labels_train)
    scores = classification_report(clf.predict(features_test),labels_test)
    print '--------------------------'*2
    print label,'\n',scores
print " "

cv = StratifiedShuffleSplit(labels, 100, random_state = 42)

### LogisticRegression ###

cv = StratifiedShuffleSplit(labels, 100, random_state = 42)
param_range=np.arange(0.00001,0.0001,0.00001)
param_grid=[{'clf__C':param_range}]
gs_Logistic = GridSearchCV(estimator=pipe2,
                   param_grid=param_grid,
                   scoring='f1',
                   cv=cv,
                   n_jobs=-1)

gs_Logistic = gs_Logistic.fit(features, labels)
pred = gs_Logistic.predict(features)
best_Logistic = gs_Logistic.best_estimator_
print classification_report(pred, labels)
print(gs_Logistic.best_score_)
print best_Logistic

best_Logistic.fit(features_train,labels_train)
pred = best_Logistic.predict(features_test)
print classification_report(pred,labels_test)


### KN ###

from sklearn.decomposition import PCA

param_grid = [{'reduce_dim__n_components' : [1,2,3,4],
              'clf__n_neighbors' : [3,5,7,9],
              'clf__weights' : ['uniform', 'distance'],
              'clf__algorithm' : ['ball_tree', 'kd_tree', 'brute','auto'],
              'clf__leaf_size' : [30, 10, 15, 20]}]

KN = KNeighborsClassifier()
pca = PCA()

pipe = Pipeline([['reduce_dim',pca],
                 ['clf',KN]])

gs_KN = GridSearchCV(estimator=pipe,
                   param_grid=param_grid,
                   scoring='f1',
                   cv=cv,
                   n_jobs=-1)



gs_KN = gs_KN.fit(features, labels)
pred = gs_KN.predict(features)
best_KN = gs_KN.best_estimator_
print classification_report(pred, labels)
print(gs_KN.best_score_)
print best_KN

best_KN.fit(features_train,labels_train)
pred = best_KN.predict(features_test)
print classification_report(pred,labels_test)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
clf = best_KN
dump_classifier_and_data(clf, my_dataset, features_list)
