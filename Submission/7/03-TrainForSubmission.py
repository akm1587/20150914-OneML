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

from sklearn import linear_model 

#CV for glm
nfold=5
candidate=np.array(10.)**np.array([-3,-2.,-1.,0.,1.,2.,3.])
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
    estimatedError[0]=temp
    counter+=1
    print("one fold done")

print(estimatedError)




#FINISH THIS PART
averageError=estimatedError.sum(axis=0)
#this is the result
#array([ 38.37318331,  55.16174402,  61.99249883,  63.41303329,
#        59.99531177,  54.5241444 ,  48.43413033])
optimalparameter=candidate[averageError==max(averageError)]



#Model
clf = linear_model.LogisticRegression(C=optimalparameter) 
clf.fit(X, Y)
#predict response
result=clf.predict(Xtest)

#convert predicted response back to original format
RESULT=np.array(map(splitResult, result.tolist()))    
#add SRId column
RESULT=np.column_stack((np.array(TestData.SRId), RESULT))
#add column name
RESULT=np.row_stack((np.array(["SRId", "CST_1_Pred", "CST_2_Pred"]), RESULT))

#np.savetxt("C:/Users/vichan/Desktop/RESULT0.txt", RESULT, fmt='%s', delimiter='\t')
np.savetxt(filepath+"RESULT0.txt", RESULT, fmt='%s', delimiter='\t')
