

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
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    clf = clf.fit(Xtrain, Ytrain)
    predictedY=clf.predict(Xtest)
    
    #test model
    error=np.mean(testY==predictedY)*100
    estimatedError.append(error)
    print("one fold done")


print(estimatedError)

