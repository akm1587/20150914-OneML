import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pickle

filepath="C:/Users/vichan/Dropbox/Documents/Microsoft/20150914-OneML/Information/"
#filepath="/Users/vc/Dropbox/Documents/Microsoft/20150914-OneML/Information/"
TrainData=pd.read_csv(filepath+"cleanedTrain0.csv")

#create bag-of-word counter for each variable
#here parameter min_df controls the minimum required appearence for words to be in dictionary

varToClean=["IST_1", "IST_2", "IST_3", "title", "problem", "error"]
#HERE I CAN TRY SMALLER MIN_DF FOR TITLE, PROBLEM AND ERROR!!!!!!!!!!!!
min_df=[1, 1, 1, 5, 5, 5]
decoder={}
result=list()

for i in range(0, len(varToClean)):
    col=varToClean[i]
    temp=TrainData[col].tolist()
    #min_df is a tuning paramter
    #When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold. 
    #This value is also called cut-off in the literature. 
    #If float, the parameter represents a proportion of documents, integer absolute counts. 
    #This parameter is ignored if vocabulary is not None.
    countMatrix=CountVectorizer(analyzer='word', min_df=min_df[i], stop_words='english', ngram_range=(1,2)) #ignore word only shown up once ever

    #some observation is empty.  This lead to problem in the fit.  So I fill in None for the empty
    for j in range(0, len(temp)):
        if(isinstance(temp[j], str)==False):
            if(np.isnan(temp[j])):
                temp[j]="none"
            
    decoder[col]=countMatrix.fit(temp)


pickle.dump(decoder, open(filepath+"decoder.pickle", "wb" ))


