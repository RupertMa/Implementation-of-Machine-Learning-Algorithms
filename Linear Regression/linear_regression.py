import numpy as np


def loadDataSet():
	data=np.loadtxt('linear-regression.txt',delimiter=",")
	X=np.ones((data.shape[0],3))
	X[:,1:]=data[:,:2]
	Y=data[:,2]
	return np.matrix(X),np.matrix(Y)

def linearRegres(X,Y):
	from numpy.linalg import inv
	W=inv(X.T*X)*X.T*Y.T
	return W

def main():
	X,Y=loadDataSet()
	W=linearRegres(X,Y)
	print "W0:", W[0]
	print "W1:", W[1]
	print "W2:", W[2]

if __name__ == "__main__":
    main()