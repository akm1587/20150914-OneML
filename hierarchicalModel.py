import numpy as np
import pandas as pd
import pickle
from sklearn import linear_model 
from sklearn.ensemble import RandomForestClassifier


#filepath="/Users/vc/Dropbox/Documents/Microsoft/20150914-OneML/Information/"
filepath="C:/Users/vichan/Dropbox/Documents/Microsoft/20150914-OneML/Information/"

# RawData=pd.read_csv(filepath+"TrainingData.tsv", sep='\t')
# hierarchy={}
# for cst1 in RawData.CST_1.unique().tolist():
#     if (cst1 in hierarchy.keys())==False:
#         hierarchy[cst1]=list()
#     for cst2 in RawData.ix[RawData.CST_1==cst1].CST_2.unique().tolist():
#         hierarchy[cst1].append(cst2)
# 
# pickle.dump(hierarchy, open(filepath+'hierarchy.pickle', "wb"))
# 
hierarchy=pickle.load(open(filepath+'hierarchy.pickle', "rb"))

class hierarchicalModel:
    def __init__(self):
        self.result={}

    def fit(self, X, Y1, Y2):
        #fit classification model on CST_1
        #model here needs to have .fit method and .predict method
        #model = linear_model.LogisticRegression() 
        model = RandomForestClassifier(random_state=123456, n_estimators=500, n_jobs=-1) 
        print('start fitting main model')
        model.fit(X, Y1)
        self.result['main']=model

        #fit model for each CST_2, which is subclass of CST_1
        for cst1 in hierarchy.keys():
            print('start fitting submodel for '+cst1)
            #find the rows in the train data that has current cst1 level
            whichRows=np.where(Y1==cst1)[0]
            #this is current data
            currentX=X[whichRows,]
            currentY=Y2.ix[Y1==cst1]

            #fit submodel
            #modelTemp = linear_model.LogisticRegression()
            modelTemp = RandomForestClassifier(random_state=123456, n_estimators=500, n_jobs=-1)
            modelTemp.fit(currentX, currentY)
            self.result[cst1]=modelTemp
        print('model fit completed')
        return()

    def predict(self, Xtest):
        #predict the CST_1, the upper hierarchy
        CST_1_Pred=self.result['main'].predict(Xtest)
        #initialize 
        CST_2_Pred=np.copy(CST_1_Pred) #since I don't know how to initialize an np.array of string, just arbitrarily choose CST1
        #given in the predicted CST_1 class, predict CST_2 #need to be super careful, if not use np.copy, CST_2_Pred is just a pointer to CST_1_Pred
        for cst1 in np.unique(CST_1_Pred).tolist():
            whichRows=np.where(CST_1_Pred==cst1)[0]
            currentX=Xtest[whichRows,]
            temp=self.result[cst1].predict(currentX)
            CST_2_Pred[whichRows]=temp
        print('all prediction completed')
        return([CST_1_Pred, CST_2_Pred])


