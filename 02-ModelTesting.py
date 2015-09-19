import pickle
import numpy as np
import pandas as pd
import scipy as sp

def bowConverter(decoder, data):
    temp=data.tolist()
    #some observation is empty.  This lead to problem in the fit.  So I fill in None for the empty
    for j in range(0, len(temp)):
        if(isinstance(temp[j], str)==False):
            if(np.isnan(temp[j])):
                temp[j]="none"
    return(decoder.transform(temp))

def splitResult(x):
    return(x.split("::$!^!$::"))


#filepath="/Users/vc/Dropbox/Documents/Microsoft/20150914-OneML/"
filepath="C:/Users/vichan/Dropbox/Documents/Microsoft/20150914-OneML/"





filepath1=filepath+"Information/"
TrainData=pd.read_csv(filepath1+"cleanedTrain.csv")
TestData=pd.read_csv(filepath1+"cleanedTest.csv")

#decoder = pickle.load( open( filepath1+"decoder.pickle", "rb" ) )
decoder = pickle.load( open( filepath1+"tfidfDecoder.pickle", "rb" ) )

varToClean=["IST_1", "IST_2", "IST_3", "title", "problem", "error"]
resultTrain=list()
resultTest=list()
for i in range(0, len(varToClean)):
    col=varToClean[i]
    resultTrain.append(bowConverter(decoder[col], TrainData[col]))
    resultTest.append(bowConverter(decoder[col], TestData[col]))
    print(resultTrain[i].shape)
    print(resultTest[i].shape)

#rf with 500 trees default setting get 60.7%
from scipy.sparse import hstack
X=hstack(resultTrain).toarray()
Xtest=hstack(resultTest).toarray()
CST_1=TrainData.CST_1
CST_2=TrainData.CST_2

nfold=5
import sys
sys.path.insert(0, filepath)
import hierarchicalModel as myfunc
from sklearn.cross_validation import StratifiedKFold
estimatedError=list()
skf = StratifiedKFold(TrainData.Y, nfold, random_state=9876543)
for train, test in skf:
    trainX=X[train,:]
    trainCST_1=CST_1[train]
    trainCST_2=CST_2[train]
    testX=X[test,:]
    testY=TrainData.Y[test]
    
    #fit model
    clf = myfunc.hierarchicalModel()
    clf.fit(trainX, trainCST_1, trainCST_2)
    predictedY0 = clf.predict(testX)
    predictedY = predictedY0[0]+"::$!^!$::"+predictedY0[1]
    
    #test model
    error=np.mean(testY==predictedY)*100
    estimatedError.append(error)
    print("one fold done")


print(estimatedError)




