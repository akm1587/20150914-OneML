def splitResult(x):
    return(x.split("::$!^!$::"))

def hybrid(HierarchicalCST1Label, HierarchicalResult, FullResult, classForHybrid):
    result=FullResult
    for c in classForHybrid:
        result[HierarchicalCST1Label==c]=HierarchicalResult[HierarchicalCST1Label==c]
    #convert predicted response back to original format
    return(np.array(map(splitResult, result.tolist())))


import cPickle as pickle
filepath="C:/Users/vichan/Dropbox/Documents/Microsoft/20150914-OneML/"
filepath1=filepath+'Information/'
hierarchy=pickle.load(open(filepath1+'hierarchy.pickle', "rb"))



import gc
import sys
sys.path.insert(0, filepath+'HierarchicalModel/rfrf')
import hierarchicalModel_rfrf as myfunc

model=pickle.load(open('E:/HierarchicalModel_rfrf.pickle', "rb"))
gc.collect()

fromTrainData={}
fromTrainData['CST_1']=model.result['main'].predict(X)
gc.collect()
fromTrainData['full']=model.predict(X)
gc.collect()


Xtest=hstack(resultTest).toarray()
gc.collect()
fromTestData={}
fromTestData['CST_1']=model.result['main'].predict(Xtest)
gc.collect()
fromTestData['full']=model.predict(Xtest)
gc.collect()


RESULT={}
RESULT['train']=fromTrainData
RESULT['test']=fromTestData

pickle(RESULT, open('','wb'))
