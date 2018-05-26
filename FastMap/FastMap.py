'''
@author Yibo Ma,Yi Chen, Kleon Loh

'''

from numpy import *

def main():
    data=loadDataSet('fastmap-data.txt')
    coordinates=fastmap(data,2)
    words=['acting','activist','compute','coward','forward','interaction','activity','odor','order','international']
    for i in range(len(words)):
    	print words[i],": ", coordinates[i]
    plotWords(coordinates)


def loadDataSet(fileName):
	with open(fileName) as f:
		fr=f.readlines()
		lst=[data.strip().split('\t') for data in fr]
		pair=[]
		for i in lst: pair.append(map(int,i))
		array=zeros((10,10))
		for i in pair:
			a=i[0]-1;b=i[1]-1
			array[b,a]=array[a,b]=float(i[2])
		return array

def pickPivot(array,k=5):
	Ob=random.randint(0, len(array)-1)
	dis=-inf
	objectID=-inf
	lst=[]
	newlst=[]
	while k:
		Oa=argmax(array[Ob])
		if [Oa,Ob] not in [entry[0:2] for entry in lst]: lst.append([Oa,Ob,Oa+Ob,array[Oa,Ob]])
		Ob=argmax(array[Oa])
		if [Oa,Ob] not in [entry[0:2] for entry in lst]: lst.append([Oa,Ob,Oa+Ob,array[Oa,Ob]])
		k-=1
	lst=mat(lst)
	Max=lst.max(axis=0)[0,-1]
	for i in lst:
		if i[0,-1]==Max:
			newlst.append(i.tolist()[0])
	newlst=sorted(newlst,key=lambda x:x[2])
	IdxOa,IdxOb=newlst[0][0:2]	
	if IdxOa>IdxOb:IdxOa,IdxOb=IdxOb,IdxOa
	return int(IdxOa),int(IdxOb)

def fastmap(array,k):
	count=0	
	X=zeros((array.shape[0],k))
	set_printoptions(suppress=True)
	while count<k:
		IdxOa,IdxOb=pickPivot(array,5)
		print IdxOa,IdxOb
		for i in range(array.shape[0]):
			X[i,count]=(array[IdxOa,i]**2+array[IdxOa,IdxOb]**2-array[i,IdxOb]**2)/(2*array[IdxOa,IdxOb])
		#print X[:,count]
		NewArray=zeros(array.shape)
		for i in range(array.shape[0]):	
			for j in range(array.shape[0]):
				Xij=X[i,count]-X[j,count]
				NewArray[i,j]=sqrt(array[i,j]**2-Xij**2)
		print NewArray
		array=NewArray.copy()
		count+=1
	return X

def plotWords(coordinates):
	import matplotlib.pyplot as plt
	word=['acting','activist','compute','coward','forward','interaction','activity','odor','order','international']
	x=coordinates[:,0]; y=coordinates[:,1]
	fig, ax = plt.subplots()
	ax.scatter(x,y)
	for i, txt in enumerate(word):
		ax.annotate(txt, (x[i],y[i]))
	plt.show()
    

if __name__ == "__main__":
    main()