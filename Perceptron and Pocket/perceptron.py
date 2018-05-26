import numpy as np

def loadDataSet():
	data=np.loadtxt("classification.txt",delimiter=",")
	X=np.ones((data.shape[0],4))
	X[:,1:]=data[:,0:3] #input values Xj for each example j
	D=data[:,3] #desired output Dj
	return X,D

def PLA(X,D,alpha=0.01):
	W=np.random.uniform(-1,1,4) #weights of each variable
	while True:
		Y=np.sum(W*X,axis=1)
		Y[np.nonzero(Y>=0)]=1.
		Y[np.nonzero(Y<0)]=-1.
		ViolatedCons=X[np.nonzero(Y!=D)]
		ViolatedD=D[np.nonzero(Y!=D)]
		if len(ViolatedCons)==0:
			break
		i=np.random.randint(len(ViolatedCons))
		W=W+alpha*ViolatedD[i]*ViolatedCons[i]
	return W

def main():
	X,D=loadDataSet()
	W=PLA(X,D)
	print "Weights are:",W

if __name__ == "__main__":
    main()
