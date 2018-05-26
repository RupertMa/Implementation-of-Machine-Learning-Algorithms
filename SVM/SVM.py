import numpy as np
from cvxopt.solvers import qp
import cvxopt

cvxopt.solvers.options['show_progress'] = False
np.set_printoptions(suppress=True)

def loadLinearSepData():
    f=np.loadtxt('linsep.txt',delimiter=',')
    features=f[:,:2]
    labels=f[:,2]
    return features,labels

def loadLinearInsepData():
    f=np.loadtxt('nonlinsep.txt',delimiter=',')
    features=f[:,:2]
    labels=f[:,2]
    return features,labels

def PolynomialKernel(Xn,X,p=2):
    return (1.+np.dot(Xn,X.T))**p    

def Linear_Kernel(Xn,X):
    return np.dot(Xn,X.T)

def RBF_Kernel(Xn,X,gamma=0.01):
    m=Xn.shape[0]
    n=X.shape[0]
    K=np.zeros((m,n))
    for i in range(n):
        for j in range(m):
            row=Xn[j,:]-X[i,:]
            K[j,i]=np.dot(row,row.T)
    K=np.exp(-gamma*K)
    return K

def SVM(features,labels,kernel=Linear_Kernel):
    n_samples=features.shape[0]
    featTfeat=kernel(features,features)
    #print(featTfeat.shape)
    Q=cvxopt.matrix(np.outer(labels,labels.T)*featTfeat)
    q=cvxopt.matrix(np.ones(n_samples)*-1)
    G=cvxopt.matrix(0.0,(n_samples,n_samples))
    G[::n_samples+1]=-1.0
    h=cvxopt.matrix(np.zeros(n_samples))
    A=cvxopt.matrix(labels,(1,n_samples))
    B=cvxopt.matrix(0.0)
    solutions=qp(P=Q,q=q,G=G,h=h,A=A,b=B)
    alpha=np.ravel(solutions['x'])
    W=np.dot((np.vstack(alpha*labels)).T,features)  #W is only meaningful when using linear kernel
    index=np.arange(len(alpha))[alpha>1e-5]
    sv=features[index,:]
    sv_labels=labels[index]
    b=1./sv_labels[0]-np.dot((alpha*labels),featTfeat[:,index[0]])
    return W,b,sv,alpha

def plotSVs(features,labels,W,b,sv,kernel):
    def f(W,b,x1,c):
        return (c-b-W[0,0]*x1)/W[0,1]
    import matplotlib.pyplot as plt
    plt.clf()
    #y_max=np.max(features[:,1])
    #y_min=np.min(features[:,1])
    x1=features[labels==1.][:,0]
    y1=features[labels==1.][:,1]
    x2=features[labels==-1.][:,0]
    y2=features[labels==-1.][:,1]
    plt.scatter(sv[:,0],sv[:,1],s=100,c='yellow')
    plt.scatter(x1,y1,c='b')
    plt.scatter(x2,y2,c='r')
    if kernel==Linear_Kernel:
        x_max=np.max(features[:,0])
        x_min=np.min(features[:,0])
        plt.plot([x_min,x_max],[f(W,b,x_min,0),f(W,b,x_max,0)],c='black')
        plt.plot([x_min,x_max],[f(W,b,x_min,1),f(W,b,x_max,1)],c='grey')
        plt.plot([x_min,x_max],[f(W,b,x_min,-1),f(W,b,x_max,-1)],c='grey')
    plt.show()

def n_fold_CV(X,D,cv=10,kernel=Linear_Kernel):
    result=[]
    num_eachfold=int(np.ceil(X.shape[0]/float(cv)))
    for i in range(cv):
        if (i+1)*num_eachfold<=X.shape[0]:
            testX=X[i*num_eachfold:(i+1)*num_eachfold,:]
            testD=D[i*num_eachfold:(i+1)*num_eachfold]
            index=range(i*num_eachfold,(i+1)*num_eachfold)
            trainD=np.delete(D,index,0)
            trainX=np.delete(X,index,0)
            W,b,sv,alpha=SVM(trainX,trainD,kernel)
            predict=np.vstack(np.dot((alpha*trainD),kernel(trainX,testX))+b).T
            predlabels=np.ones((1,testX.shape[0]))
            predlabels[predict<0.]=-1.
            result.append(np.sum(predlabels==testD)/float(testX.shape[0]))
        else:
            testX=X[i*num_eachfold:,:];testD=D[i*num_eachfold:]
            trainX=X[:i*num_eachfold,:];trainD=D[:i*num_eachfold]
            W,b,sv,alpha=SVM(trainX,trainD,kernel)
            predict=np.vstack(np.dot((alpha*trainD),kernel(trainX,testX))+b).T
            predlabels=np.ones((1,testX.shape[0]))
            predlabels[predict<0.]=-1.
            result.append(np.sum(predlabels==testD)/float(testX.shape[0]))
    return np.average(result)


def main():
    features,labels=loadLinearSepData()
    W,b,sv,alpha=SVM(features,labels)
    accuracy=n_fold_CV(features,labels,cv=10)
    print('++++++++++Linear Separable Data++++++++++')
    print('The equation of the line of separation: %.2fx1+%.2fx2+%.2f=0' %(W[0,0],W[0,1],b))
    print('Accuracy: %.2f%%' %(accuracy*100))
    plotSVs(features,labels,W,b,sv,kernel=Linear_Kernel)
    
    features,labels=loadLinearInsepData()
    W,b,sv,alpha=SVM(features,labels,kernel=RBF_Kernel)
    accuracy=n_fold_CV(features,labels,cv=10,kernel=RBF_Kernel)
    print('++++++++++Linear Inseparable Data with RBF Kernel++++++++++')
    print('Alpha:',alpha)
    print('Intercept:',b)
    print('Accuracy: %.2f%%' %(accuracy*100))
    plotSVs(features,labels,W,b,sv,kernel=RBF_Kernel)
    
    W,b,sv,alpha=SVM(features,labels,kernel=PolynomialKernel)
    accuracy=n_fold_CV(features,labels,cv=10,kernel=PolynomialKernel)
    print('++++++++++Linear Inseparable Data with Polynomial Kernel++++++++++')
    print('Alpha:',alpha)
    print('Intercept:',b)
    print('Accuracy: %.2f%%' %(accuracy*100))
    plotSVs(features,labels,W,b,sv,kernel=PolynomialKernel)

if __name__=="__main__":
    main()
