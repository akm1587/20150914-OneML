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

from sklearn.ensemble import GradientBoostingClassifier 

#Model
result = GradientBoostingClassifier(verbose=1, random_state=123456, n_estimators=2000, learning_rate=0.1, max_features=75, max_depth=5)
result.fit(X, Y)

#save model for future use
pickle.dump(result, open(filepath+'DirectModel/gbm.pickle', 'wb'))


#predict response
predictedy=result.predict(Xtest)
#convert predicted response back to original format
RESULT=np.array(map(splitResult, predictedy.tolist()))    
#add SRId column
RESULT=np.column_stack((np.array(TestData.SRId), RESULT))
#add column name
RESULT=np.row_stack((np.array(["SRId", "CST_1_Pred", "CST_2_Pred"]), RESULT))

#np.savetxt("C:/Users/vichan/Desktop/RESULT0.txt", RESULT, fmt='%s', delimiter='\t')
np.savetxt(filepath1+"RESULT0.txt", RESULT, fmt='%s', delimiter='\t')


