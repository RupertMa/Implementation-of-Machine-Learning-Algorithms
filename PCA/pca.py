'''
@authro Yibo Ma, Kleon Loh, Yi Chen

'''

from numpy import *

def loadDataSet(fileName, delim='\t'):
    with open(fileName) as f:
        fr=f.readlines()
        dataArr=[entry.rstrip().split(delim) for entry in fr]
        dataMat=mat([map(float,data) for data in dataArr])
        return dataMat

def pca(dataMat,k=999999):
    meanVals=mean(dataMat,axis=0)
    dataMat=dataMat-meanVals
    covMat=dataMat.T*dataMat/float(len(dataMat))
    #print covMat
    w,v=linalg.eig(covMat)
    #print w,v
    eigenVIndex=argsort(w)[:-(k+1):-1]
    sortedV=v[:,eigenVIndex]
    return sortedV,sortedV.T*dataMat.T

def plot2D(Mat):
    import matplotlib.pyplot as plt
    plt.scatter(x=Mat[0],y=Mat[1])
    plt.show()


def main():
    data=loadDataSet("pca-data.txt")
    eigenVector,lowDData=pca(data,k=2)
    print "======the directions of the first two principal components======"
    print eigenVector
    print "==================2D coordinates of the data===================="
    print lowDData
    plot2D(lowDData)

if __name__ == "__main__":
    main()

