import cPickle as pickle
import gc
import sys
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


filepath="C:/Users/vichan/Dropbox/Documents/Microsoft/20150914-OneML/"
filepath1=filepath+'Information/'
#hierarchy=pickle.load(open(filepath1+'hierarchy.pickle', "rb"))
#sys.path.insert(0, filepath+'DirectModel/')
#import glm as myfunc

#decoder = pickle.load( open( filepath1+"decoder.pickle", "rb" ) )
decoder = pickle.load( open( filepath1+"tfidfDecoder.pickle", "rb" ) )

model=pickle.load(open(filepath+"DirectModel/glm/glm.pickle", "rb"))
TrainData=pd.read_csv(filepath1+"cleanedTrain.csv")
TestData=pd.read_csv(filepath1+"cleanedTest.csv")


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
Xtrain=hstack(resultTrain).toarray()
Xtest=hstack(resultTest).toarray()


trainPredict=model.predict(Xtrain)
testPredict=model.predict(Xtest)
result={}
result['train']=trainPredict
result['test']=testPredict
pickle.dump(result, open(filepath+"DirectModel/glmPrediction.pickle", "wb"))
