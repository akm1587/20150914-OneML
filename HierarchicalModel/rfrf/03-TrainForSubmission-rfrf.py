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


#filepath="/Users/vc/Dropbox/Documents/Microsoft/20150914-OneML/"
filepath="C:/Users/vichan/Dropbox/Documents/Microsoft/20150914-OneML/"

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
CST_1=np.array(TrainData.CST_1.tolist())
CST_2=np.array(TrainData.CST_2.tolist())


import sys
sys.path.insert(0, filepath+'HierarchicalModel/')
import hierarchicalModel_rfrf as myfunc


#Model
#fit model
result = myfunc.hierarchicalModel()
result.fit(X, CST_1, CST_2)


#pickle.dump(result, open(filepath+'HierarchicalModel/HierarchicalModel_rfrf.pickle', 'wb'))
#this thing is 50+GB, too large to write to dropbox, so write it to desktop
pickle.dump(result, open('C:/Users/vichan/Desktop/HierarchicalModel_rfrf.pickle', 'wb'))
