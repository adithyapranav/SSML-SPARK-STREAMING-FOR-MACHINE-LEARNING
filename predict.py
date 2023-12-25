import pickle
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


file=open('/home/pes2ug19cs097/Downloads/BNB.pickle','rb')
BNB=pickle.load(file)

file=open('/home/pes2ug19cs097/Downloads/MNB.pickle','rb')
MNB=pickle.load(file)

file=open('/home/pes2ug19cs097/Downloads/clf.pickle','rb')
clf=pickle.load(file)

file=open('/home/pes2ug19cs097/Downloads/kmeans.pickle','rb')
kmeans=pickle.load(file)

BNB_txt=open('/home/pes2ug19cs097/Downloads/BNB','a')
MNB_txt=open('/home/pes2ug19cs097/Downloads/MNB','a')
clf_txt=open('/home/pes2ug19cs097/Downloads/clf','a')
km_txt=open('/home/pes2ug19cs097/Downloads/kmeans','a')

BNB_array=[[0,0],[0,0]]
MNB_array=[[0,0],[0,0]]
clf_array=[[0,0],[0,0]]
km_array=[[0,0],[0,0]]



def prediction(rdd):
	x=rdd.select('features').rdd.flatMap(lambda x: x).collect()
	y=rdd.select('Sentiment').rdd.flatMap(lambda x: x).collect()

	BNB_accuracy= accuracy_score(BNB.predict(x),y) 
	BNB_txt.write(str(BNB_accuracy))
	BNB_txt.write('\n')
	array= confusion_matrix(y, BNB.predict(x))
	
	for i in range(2):
		for j in range(2):
			BNB_array[i][j]+= array[i][j].item()

	MNB_accuracy= accuracy_score(MNB.predict(x),y)
	MNB_txt.write(str(MNB_accuracy))
	MNB_txt.write('\n')
	array= confusion_matrix(y, MNB.predict(x))
	
	for i in range(2):
		for j in range(2):
			MNB_array[i][j]+= array[i][j].item()

	clf_accuracy= accuracy_score(clf.predict(x),y) 
	clf_txt.write(str(clf_accuracy))
	clf_txt.write('\n')
	array= confusion_matrix(y, clf.predict(x))

	for i in range(2):
		for j in range(2):
			clf_array[i][j]+= array[i][j].item()
			
	kmeans_accuracy= accuracy_score(clf.predict(x),y) 
	km_txt.write(str(kmeans_accuracy))
	km_txt.write('\n')
	array= confusion_matrix(y, kmeans.predict(x))

	for i in range(2):
		for j in range(2):
			km_array[i][j]+= array[i][j].item()

def accuracy():
	
	print()
	tot=0
	corr=0
	for i in range(2):
		for j in range(2):
			tot+=BNB_array[i][j]
			if(i==j):
				corr+=BNB_array[i][j]
	print('BNB Accuracy:',corr/tot)
	
	tot=0
	corr=0
	for i in range(2):
		for j in range(2):
			tot+=MNB_array[i][j]
			if(i==j):
				corr+=MNB_array[i][j]
	print('MNB Accuracy:',corr/tot)
	
	tot=0
	corr=0
	for i in range(2):
		for j in range(2):
			tot+=clf_array[i][j]
			if(i==j):
				corr+=clf_array[i][j]
				
	print('SGD Accuracy:',corr/tot)
	
	tot=0
	corr=0
	for i in range(2):
		for j in range(2):
			tot+=km_array[i][j]
			if(i==j):
				corr+=km_array[i][j]
				
	print('Kmeans Accuracy:',corr/tot)
	print()



