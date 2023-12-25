from sklearn import linear_model
from sklearn.naive_bayes import BernoulliNB, MultinomialNB

from sklearn.cluster import MiniBatchKMeans	
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pickle
import os
import joblib

BNB=BernoulliNB()
MNB=MultinomialNB()
clf = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)
kmeans= MiniBatchKMeans(n_clusters=2)

def model_training(rdd):
		
	x=rdd.select('features').rdd.flatMap(lambda x: x).collect()
	y=rdd.select('Sentiment').rdd.flatMap(lambda x: x).collect()
	#x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1)
	
	print()
	BNB.partial_fit(x,y,classes=np.unique(y))
	#accuracy_bnb=accuracy_score(BNB.predict(x_test),y_test) 
	#print('BernoulliNB: ',accuracy_bnb)
	print('BernoulliNB Model is trained')
	
	MNB.partial_fit(x,y,classes=np.unique(y))
	#accuracy_mnb=accuracy_score(MNB.predict(x_test),y_test)
	#print('MultinomialNB: ',accuracy_mnb)
	print('MultinomialNB Model is trained')
	
	clf.partial_fit(x, y, classes=np.unique(y))
	#accuracy_sgd=accuracy_score(clf.predict(x_test),y_test)
	#print('SGDClassifier: ',accuracy_sgd)	
	print('SGDClassifier Model is trained')
	
	kmeans.partial_fit(x, y)
	#accuracy_kmeans=accuracy_score(clf.predict(x_test),y_test)
	#print('KmeansClassifier: ',accuracy_kmeans)
	print('Kmeans Model is trained')
	print()	

	here = os.path.dirname(os.path.abspath(__file__))
	with open(os.path.join(here, "BNB.pickle"), "wb") as f:
		pickle.dump(BNB, f)
	
	with open(os.path.join(here, "MNB.pickle"), "wb") as f:
		pickle.dump(MNB, f)

	with open(os.path.join(here, "clf.pickle"), "wb") as f:
		pickle.dump(clf, f)
		
	with open(os.path.join(here, "kmeans.pickle"), "wb") as f:
		pickle.dump(kmeans, f)
