import treePlotter
import trees
from math import log


#Decision Tree Algorithm

def attributeScan(dataset,attributes):
	dic={}
	for i in range(len(dataset[0])-1):
		newentry=list()
		values=[entry[i] for entry in dataset]
		uniqueVals=list(set(values))
		dic[attributes[i]]=uniqueVals
	return dic


def splitDataset(attributeVal,attributeIndex,dataset):
	subdataset=[]
	for entry in dataset:
		if entry[attributeIndex]==attributeVal:
			newentry=entry[:]
			del newentry[attributeIndex]
			subdataset.append(newentry)
	return subdataset


def calEntropy(dataset):
	numEntry=len(dataset)
	Entropy=0.0
	lst=[entry[-1] for entry in dataset]
	uniquevalue=set(lst)
	for value in uniquevalue:
		p=float(lst.count(value))/numEntry
		Entropy-=p*log(p,2)
	return Entropy

def chooseBestAttribute(attributes,dataset):
	Entropy=calEntropy(dataset)
	numEntry=len(dataset)
	BestAttribute=''; MaxInfoGain=-0.1
	for Index in range(len(dataset[0])-1):
		NewEntropy=0.0
		lst=[entry[Index] for entry in dataset]
		uniqueVal=set(lst)
		for value in uniqueVal:
			subset=splitDataset(value,Index,dataset)
			prob=float(len(subset))/numEntry
			NewEntropy+=prob*calEntropy(subset)
		if (Entropy-NewEntropy)>MaxInfoGain: 
			MaxInfoGain=Entropy-NewEntropy
			BestAttribute=attributes[Index]
			AttributeIndex=Index
	return BestAttribute

def MajorityCount(dataset):
	labelList=[entry[-1] for entry in dataset]
	dic={}
	#print labelList
	for label in labelList:
		dic[label]=dic.get(label,0)+1
	num_label=sorted(dic.items(),key=lambda label:label[1],reverse=True)
	if num_label[0][1]==num_label[1][1]:
		return 'Tie'
	else:
		return num_label[0][0]


def createTree(dataset,attributes,attribute_dic):
	labelList=[entry[-1] for entry in dataset]
	if len(dataset[0])==1:
		label=MajorityCount(dataset)
		return label
	if labelList.count(labelList[0])==len(dataset):
		return labelList[0]
	BestAttribute=chooseBestAttribute(attributes,dataset)
	attributeIndex=attributes.index(BestAttribute)
	tree={BestAttribute:{}}
	del attributes[attributeIndex]
	attributeVals=set([entry[attributeIndex] for entry in dataset])
	if len(attributeVals)!=len(attribute_dic[BestAttribute]):
		for AttributeValue in attribute_dic[BestAttribute]:
			if AttributeValue not in attributeVals:
				tree[BestAttribute][AttributeValue]='Tie'
	for attributeVal in attributeVals:
		subattributes=attributes[:]
		tree[BestAttribute][attributeVal]=createTree(splitDataset(attributeVal,attributeIndex,dataset),subattributes,attribute_dic)
	return tree

def classify(tree,dataset,attributes):
	for attributeName in tree.keys():
		attributeVals=tree[attributeName]
		for attributeVal in attributeVals.keys():
			if attributeVal==dataset[attributes.index(attributeName)]:
				if type(attributeVals[attributeVal]).__name__!='dict':
					return attributeVals[attributeVal]
				else:
					return classify(attributeVals[attributeVal],dataset,attributes)


#Data Preprocessing
fhand=open('dt-data.txt','r').read().splitlines()

attributes=fhand[0][1:][:-1].split(',')
newatt=[]
for attribute in attributes[:len(attributes)-1]:
	attribute=attribute.strip()
	newatt.append(attribute)
attributes=newatt

dataset=fhand[2:]
dataset2=[]
for entry in dataset:
	newentry=list()
	entry=entry[3:][:-1].split(',')
	for item in entry:	
		item=item.strip()
		newentry.append(item)
	dataset2.append(newentry)

# Execute code on dataset 
attribute_dic=attributeScan(dataset2,attributes[:])
tree=createTree(dataset2,attributes[:],attribute_dic)
print 'Tree',tree #print the decision tree that is produced
dataEntry=['Moderate','Cheap','Loud','City-Center','No','No']
print 'Prediction Result:', classify(tree,dataEntry,attributes[:]) #Make prediction
treePlotter.createPlot(tree)  #visualize tree to make checking easier 


