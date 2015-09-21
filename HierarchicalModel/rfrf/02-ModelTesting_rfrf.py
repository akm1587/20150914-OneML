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
CST_1=np.array(TrainData.CST_1.tolist())
CST_2=np.array(TrainData.CST_2.tolist())
Y=np.array(TrainData.Y.tolist())

import scipy as sp

nfold=5
import sys
sys.path.insert(0, filepath+'HierarchicalModel/')
import hierarchicalModel_rfrf as myfunc
from sklearn.cross_validation import StratifiedKFold
estimatedError1=list()
estimatedError2=list()
estimatedError3=list()
skf = StratifiedKFold(y=TrainData.Y, n_folds=nfold, random_state=987654)
for train, test in skf:
    trainX=X[train,:]
    trainCST_1=CST_1[train]
    trainCST_2=CST_2[train]
    testX=X[test,:]
    testY=Y[test]
    
    #fit model
    clf = myfunc.hierarchicalModel()
    clf.fit(trainX, trainCST_1, trainCST_2)
    CST_1_Pred, CST_2_Pred = clf.predict(testX)
    predictedY = np.array(['%s::$!^!$::%s' % t for t in zip(CST_1_Pred.tolist(), CST_2_Pred.tolist())])
    
    #test model
    error1=np.mean(CST_1[test]==np.array(CST_1_Pred))*100
    error2=np.mean(testY==predictedY)*100
    estimatedError1.append(error1)
    estimatedError2.append(error2)
    
    temp=list()
    for k in TrainData.CST_1.unique().tolist():
        temp.append(np.mean(testY[CST_1_Pred==k]==predictedY[CST_1_Pred==k])*100)
    estimatedError3.append(temp)

    print("one fold done")
estimatedError3=np.array(estimatedError3)



print(estimatedError1)
print(estimatedError2)
print(estimatedError3)




