import pandas as pd
from stemming.porter2 import stem
import numpy as np
param1='Train'
#param1='Test'
#filepath="/Users/vc/Dropbox/Documents/Microsoft/20150914-OneML/Information/"
filepath="C:/Users/vichan/Dropbox/Documents/Microsoft/20150914-OneML/Information/"


############################################
#load data
############################################
RawData=pd.read_csv(filepath+param1+"ingData.tsv", sep='\t')


############################################
#data exploration
############################################
print("Size of data")
print(RawData.shape)

#print("frequency of CTS_1")
#print(RawData.groupby('CST_1').size())

#print("frequency of IST_1")
#print(RawData.groupby(('IST_1')).size())

#print("frequency of CST_2")
#print(RawData.groupby('CST_2').size())

#print("frequency of IST_2")
#print(RawData.groupby('IST_2').size())

RawData.head()


############################################
#data basic text cleaning
############################################

varToClean=["IST_1", "IST_2", "IST_3", "title", "problem", "error"]

for col in varToClean:
    #lower case everything
    RawData[col]=RawData[col].str.lower()
    
    #replace abbreviation to full word
    RawData[col]=RawData[col].str.replace("can't", "cannot")
    RawData[col]=RawData[col].str.replace("couldn't", "could not")
    RawData[col]=RawData[col].str.replace("hasn't", "has not")
    RawData[col]=RawData[col].str.replace("haven't", "have not")
    RawData[col]=RawData[col].str.replace("wasn't", "was not")
    RawData[col]=RawData[col].str.replace("weren't", "were not")
    RawData[col]=RawData[col].str.replace("won't", "will not")
    RawData[col]=RawData[col].str.replace("didn't", "did not")
    RawData[col]=RawData[col].str.replace("doesn't", "does not")
    RawData[col]=RawData[col].str.replace("don't", "do not")
    RawData[col]=RawData[col].str.replace("'ve", " have")
    RawData[col]=RawData[col].str.replace("i'm", "I am")
    
    #all of these are equivalent to sing in
    RawData[col]=RawData[col].str.replace("sign-in", "signin")
    RawData[col]=RawData[col].str.replace("sign in", "signin")
    RawData[col]=RawData[col].str.replace("sign-on", "signin")
    RawData[col]=RawData[col].str.replace("signon", "signin")
    RawData[col]=RawData[col].str.replace("sign on", "signin")

    #repalce punctional, speicla characters with space
    #only keep a to z and 0 to 9, replace all others with space
    RawData[col]=RawData[col].str.replace("[^a-z0-9]", " ")
    #replace multiple spaces with one space
    RawData[col]=RawData[col].str.replace(" +", " ")

    def tempfn(x):
        #some observation is empty.  This lead to problem in the fit.  So I fill in None for the empty
        if(isinstance(x, str)==False):
            if(np.isnan(x)):
                x="none"
        temp=x.split(' ')
        result=map(stem, temp)
        return(' '.join(result))

    RawData[col]=map(tempfn, RawData[col].tolist())




if param1=="Train":
    Y=RawData.CST_1+"::$!^!$::"+RawData.CST_2
    RawData['Y']=Y
    #since SRId is unique for each obs, remove it
    RawData=RawData.drop('SRId', axis=1)

############################################
#output cleaned data
############################################
RawData.to_csv(filepath+"cleaned"+param1+".csv", index=False)
