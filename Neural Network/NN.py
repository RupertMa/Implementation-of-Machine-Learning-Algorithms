# -*- coding: utf-8 -*-
import numpy as np

def imgProcess(img):
    from skimage import io
    img = io.imread(img)
    img=(img/255.).flatten()
    return img

def loadImages(filename):
    labels=[]
    with open(filename) as f:
        imgs=f.readlines()
        for img in imgs:
            img='./'+img.rstrip()
            if 'down' in img:
                labels.append(1)
            else:
                labels.append(0)
            if 'dataset' not in locals():
                dataset=imgProcess(img)
            else:
                dataset=np.vstack((dataset,imgProcess(img)))
        return dataset,labels

def sigmoid(s):
    return 1./(1+np.exp(-s))

def hiddenLayer(instance,w):
    s=np.dot(instance,w)     # instance shape (1,960)
    Xj=sigmoid(s)
    return Xj,s

def feedForward(instance,w2,w3):
    instance=np.vstack(instance).T   #instance shape(1，960）
    s2=np.dot(instance,w2)     
    Xj2=sigmoid(s2)     # Xj2 shape (1,100)
    s3=np.dot(Xj2,w3)   #s3 shape (1,1)
    Xj3=sigmoid(s3)
    return Xj2,Xj3,instance

    
def backPropagation(Xj2,Xj3,w3,y):
    delta3=2.*(Xj3-y)*Xj3*(1-Xj3) #delta3 shape (1,1)
    delta2=(Xj2*(1-Xj2))*(delta3*w3.T)   #delta2 shape(1,100)
    return delta2,delta3

def checkAccuracy(train,trainLabels,w2,w3):
    count=0.
    for index,instance in enumerate(train):
        instance=np.vstack(instance).T #instance shape (1,960) w2 (960,100)
        s2=np.dot(instance,w2) #Xj2 shape (1,100)
        Xj2=sigmoid(s2)
        s3=np.dot(Xj2,w3)
        Xj3=sigmoid(s3)          
        if Xj3>=0.5:
            Xj3=1
        else:
            Xj3=0
        if Xj3==trainLabels[index]:
            count+=1
    return count/train.shape[0]

def classify(w2,w3):
    count=0.
    test,testLabels=loadImages('downgesture_test.list')
    f=open('downgesture_test.list').readlines()
    filenames=[i.rstrip() for i in f]
    testLabels=np.vstack(testLabels)
    for index,instance in enumerate(test):
        instance=np.vstack(instance).T #instance shape (1,960) w2 (960,100)
        s2=np.dot(instance,w2) #Xj2 shape (1,100)
        Xj2=sigmoid(s2)
        s3=np.dot(Xj2,w3)
        Xj3=sigmoid(s3)    
        #error+=(Xj3-testLabels[index])**2        
        if Xj3>=0.5:
            Xj3=1
            print filenames[index],testLabels[index],'Predict: 1'
        else:
            Xj3=0
            print filenames[index],testLabels[index],'Predict: 0'
        if Xj3==testLabels[index]:
            count+=1
    return "Accuracy: ",count/test.shape[0]

def NN(eta=0.1,num_epoch=1000,num_perceptrons=100):
    accuracy_best=0.
    train,trainLabels=loadImages('downgesture_train.list')   #train shape(184,960)
    trainLabels=np.vstack(trainLabels)  #trainLabels shape(184,1)
    i=train.shape[1]
    w2=np.random.uniform(-1,1,(i,num_perceptrons))  #w2 shape(960,100)
    w3=np.random.uniform(-1,1,(num_perceptrons,1))  #w3 shape(100,1)
    for iteration in range(num_epoch):
        for index,instance in enumerate(train):
            Xj2,Xj3,instance=feedForward(instance,w2,w3)
            delta2,delta3=backPropagation(Xj2,Xj3,w3,trainLabels[index])            
            w3=w3-eta*Xj2.T*delta3
            w2=w2-eta*instance.T*delta2   #w2 shape(961,100)
        accuracy=checkAccuracy(train,trainLabels,w2,w3)
        #print iteration,accuracy
        if accuracy==1.:
            w2_best,w3_best=w2,w3
            break
        else:
            if accuracy>accuracy_best:
                w2_best,w3_best=w2,w3
    return w2_best,w3_best    

def main():
    import time
    startime=time.clock()
    w2_best,w3_best=NN()
    print classify(w2_best,w3_best)
    print "Runtime: ",time.clock()-startime,'seconds.'

if __name__ == "__main__":
    main()