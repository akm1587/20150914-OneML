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




#filepath="/Users/vc/Dropbox/Documents/Microsoft/20150914-OneML/Information/"
filepath="C:/Users/vichan/Dropbox/Documents/Microsoft/20150914-OneML/Information/"
TrainData=pd.read_csv(filepath+"cleanedTrain0.csv")
TestData=pd.read_csv(filepath+"cleanedTest0.csv")

decoder = pickle.load( open( filepath+"decoder1.pickle", "rb" ) )

varToClean=["IST_1", "IST_2", "IST_3", "title", "problem", "error"]
resultTrain=list()
resultTest=list()
for i in range(0, len(varToClean)):
    col=varToClean[i]
    resultTrain.append(bowConverter(decoder[col], TrainData[col]))
    resultTest.append(bowConverter(decoder[col], TestData[col]))
    print(resultTrain[i].shape)
    print(resultTest[i].shape)

#rf with 1000 trees default setting get 59.5%
from scipy.sparse import hstack
X=hstack(resultTrain)
Xtest=hstack(resultTest)
Y=TrainData.Y

#SimpleModel
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
clf = clf.fit(X, Y)

result=clf.predict(Xtest)

def tempfn(x):
    return(x.split("::$!^!$::"))

RESULT=np.array(map(tempfn, result.tolist()))    
RESULT=np.column_stack((np.array(TestData.SRId), RESULT))

RESULT=np.row_stack((np.array(["SRId", "CST_1_Pred", "CST_2_Pred"]), RESULT))
np.savetxt("C:/Users/vichan/Desktop/RESULT0.txt", RESULT, fmt='%s', delimiter='\t')
