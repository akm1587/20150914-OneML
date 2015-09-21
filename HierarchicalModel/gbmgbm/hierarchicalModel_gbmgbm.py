import numpy as np
import pandas as pd
import pickle
from sklearn import linear_model 
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.cross_validation import StratifiedKFold
import scipy as sp


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
        model = GradientBoostingClassifier(verbose=1, random_state=123456, n_estimators=1500, learning_rate=0.1, max_features=75, max_depth=5)
        #model = CVmodel(10.**np.arange(-1.5, 1.5, 0.25), 5) 
        print('start fitting CST_1 level model')
        model.fit(X, Y1)
        self.result['main']=model

        #fit model for each CST_2, which is subclass of CST_1
        for cst1 in hierarchy.keys():
            print('start fitting submodel for '+cst1)
            #find the rows in the train data that has current cst1 level
            whichRows=np.where(Y1==cst1)[0]
            #this is current data
            currentX=X[whichRows,]
            currentY=Y2[Y1==cst1]

            #fit submodel
            #modelTemp = linear_model.LogisticRegression()
            modelTemp = GradientBoostingClassifier(verbose=1, random_state=123456, n_estimators=2000, learning_rate=0.1, max_features=75, max_depth=5)
            #modelTemp = CVmodel(10.**np.arange(-1.5, 1.5, 0.25), 5) 
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

class CVmodel:
    def __init__(self, candidate, nfold=5):
        self.candidate=candidate
        self.nfold=nfold

    def fit(self, X, Y):
        estimatedError=np.empty((self.nfold, self.candidate.size))
        skf = StratifiedKFold(y=Y, n_folds=self.nfold, random_state=13579)
        counter=0
        for train, test in skf:
            trainX=X[train,:]
            trainY=Y[train]
            testX=X[test,:]
            testY=Y[test]

            temp=list()
            for i in range(0, self.candidate.size):
                #fit model
                logreg = linear_model.LogisticRegression(C=self.candidate[i]) 
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
        self.CVresults=estimatedError

        #FINISH THIS PART
        averageError=estimatedError.mean(axis=0)
        print(averageError)
        self.optimalparameter=self.candidate[averageError==max(averageError)]
        print('CV completed')

        self.model = linear_model.LogisticRegression(C=self.optimalparameter) 
        self.model.fit(X, Y)

    def predict(self, X):
        return(self.model.predict(X))
