import matplotlib.pyplot as plt

x_axis= list(range(16))

def for_graph1():
	BNB=[]
	MNB= []
	clf= []
	with open('BNB','r') as f:
		fp= f.readlines()
		for i in fp:
			BNB.append(i.strip())
			
	with open('MNB','r') as f:
		fp= f.readlines()
		for i in fp:
			MNB.append(i.strip())
			
	with open('clf','r') as f:
		fp= f.readlines()
		for i in fp:
			clf.append(i.strip())	
	
	
	plt.plot(x_axis,BNB, label="BNB Plot")
	plt.plot(x_axis,MNB, label="MNB Plot")
	plt.plot(x_axis,clf, label="SGD Plot")
	plt.xlabel('Batch Number')
	plt.ylabel('Accuracy')
	plt.title('Graph')
	plt.legend()
	plt.show()
for_graph1()	

def for_graph2():
	acc=[]
	model_list=['BNB','MNB','SGD']
	
	with open('clf','r') as f:
		fp= f.readlines()[-1]
		acc.append(fp)	

			
	with open('MNB','r') as f:
		fp= f.readlines()[-1]
		acc.append(fp)
		
	with open('BNB','r') as f:
		fp= f.readlines()[-1]
		acc.append(fp)	
		print(fp)		
	
	fig=plt.figure(figsize=(10,10))
	plt.bar(model_list, acc, color='maroon', width=0.4)
	plt.xlabel('Model')
	plt.ylabel('Accuracy')
	plt.title('Graph')
	plt.show()
	
for_graph2()
