import numpy as np

def loadDataSet():
	data=np.loadtxt("classification.txt",delimiter=",")
	X=np.ones((data.shape[0],4))
	X[:,1:]=data[:,0:3] #input values Xj for each example j
	D=data[:,4] #desired output Dj
	return np.matrix(X),np.matrix(D)

def sigmoid(s):
	return 1./(1+np.exp(-s))

def deriv_helper(s): 
	return 1./(1+np.exp(s))

def gradDescent(X, D, eta=0.001, iteration=7000):
	n,d = np.shape(X)
	weights = np.ones((1,d))
	for k in range(iteration):
		s = np.multiply(D, weights*X.T) 
		deltaEin = (np.multiply(deriv_helper(s),D)*X*-1.) / n # Calcualte the derivative of cost function.
		weights = weights - eta * deltaEin
	return weights

def n_fold_CV(X,D,cv=5):
	result=[]
	num_eachfold=int(np.ceil(X.shape[0]/float(cv)))
	for i in range(cv):
		if (i+1)*num_eachfold<=X.shape[0]:
			testX=X[i*num_eachfold:(i+1)*num_eachfold,:]
			testD=D[:,i*num_eachfold:(i+1)*num_eachfold]
			index=range(i*num_eachfold,(i+1)*num_eachfold)
			trainD=np.delete(D,index,1);trainX=np.delete(X,index,0)
			W=gradDescent(trainX,trainD)
			Y=sigmoid(W*testX.T)
			Y[np.nonzero(Y>=0.5)]=1.
			Y[np.nonzero(Y<0.5)]=-1.
			result.append(float(np.sum((Y==testD),axis=1)/float(testD.shape[1])))
		else:
			testX=X[i*num_eachfold:,:];testD=D[:,i*num_eachfold:]
			trainX=X[:i*num_eachfold,:];trainD=D[:,:i*num_eachfold]
			W=gradDescent(trainX,trainD)
			Y=sigmoid(W*testX.T)
			Y[np.nonzero(Y>=0.5)]=1.
			Y[np.nonzero(Y<0.5)]=-1.
			result.append(np.sum((Y==testD),axis=1)/float(testD.shape[1]))
	return result


def main():
	X,D=loadDataSet()
	W=gradDescent(X,D)
	print "Weights are:",W
	results=n_fold_CV(X,D)
	print "Accuracy in each run:",results
	print "Accuracy: %0.2f (+/- %0.2f)" % (np.mean(results), np.std(results) * 2)
	

if __name__ == "__main__":
    main()



