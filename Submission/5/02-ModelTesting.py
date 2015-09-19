import pickle
import numpy as np
import pandas as pd


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



#filepath="/Users/vc/Dropbox/Documents/Microsoft/20150914-OneML/Information/"
filepath="C:/Users/vichan/Dropbox/Documents/Microsoft/20150914-OneML/Information/"
TrainData=pd.read_csv(filepath+"cleanedTrain.csv")
TestData=pd.read_csv(filepath+"cleanedTest.csv")

#decoder = pickle.load( open( filepath+"decoder.pickle", "rb" ) )
decoder = pickle.load( open( filepath+"tfidfDecoder.pickle", "rb" ) )

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
Y=TrainData.Y

nfold=5
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedKFold
estimatedError=list()
skf = StratifiedKFold(TrainData.Y, nfold)
for train, test in skf:
    trainX=X[train,:]
    trainY=Y[train]
    testX=X[test,:]
    testY=Y[test]

    #fit model
    clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
    clf = clf.fit(trainX, trainY)
    predictedY=clf.predict(testX)
    
    #test model
    error=np.mean(testY==predictedY)*100
    estimatedError.append(error)
    print("one fold done")


print(estimatedError)




