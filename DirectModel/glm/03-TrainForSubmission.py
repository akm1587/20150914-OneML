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



filepath="/Users/vc/Dropbox/Documents/Microsoft/20150914-OneML/"
#filepath="C:/Users/vichan/Dropbox/Documents/Microsoft/20150914-OneML/"




filepath1=filepath+'Information/'
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
Y=TrainData.Y

from sklearn import linear_model 

#CV for glm
nfold=5
candidate=np.array(10.)**np.arange(-1.5,1.5,0.25)
#from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model 
from sklearn.cross_validation import StratifiedKFold
estimatedError=np.empty((nfold, candidate.size))
skf = StratifiedKFold(TrainData.Y, nfold)
counter=0
for train, test in skf:
    trainX=X[train,:]
    trainY=Y[train]
    testX=X[test,:]
    testY=Y[test]
    
    temp=list()
    for i in range(0, candidate.size):
        #fit model
        logreg = linear_model.LogisticRegression(C=candidate[i]) 
        logreg = logreg.fit(trainX, trainY)
        predictedY=logreg.predict(testX)
#    clf = RandomForestClassifier(n_estimators=500, n_jobs=7)
#    clf = clf.fit(trainX, trainY)
#    predictedY=clf.predict(testX)
    
    #test model
        error=np.mean(testY==predictedY)*100
        temp.append(error)
    estimatedError[counter]=temp
    counter+=1
    print("one fold done")

print(estimatedError)




#FINISH THIS PART
averageError=estimatedError.sum(axis=0)
print(averageError)
#this is the result
#array([ 38.37318331,  55.16174402,  61.99249883,  63.41303329,
#        59.99531177,  54.5241444 ,  48.43413033])
optimalparameter=candidate[averageError==max(averageError)]



#Model
result = linear_model.LogisticRegression(C=optimalparameter)
result.fit(X, Y)
pickle.dump(result, open(filepath+'DirectModel/glm.pickle', 'wb'))

