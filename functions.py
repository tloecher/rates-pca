/// Test ///
%load_ext autoreload
%autoreload 2

from pipe import getdata
from pandas.tseries.offsets import BDay

from IPython.core.debugger import Tracer #pdb.set_trace()
import pdb

import statsmodels.tsa.stattools as ts

import random
import time
from itertools import compress
from sklearn.decomposition import PCA

## Import data from csv files ##

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def csv2df(csvName):
    
    # Import and format CSV as df
    # See spotEUR.csv for formatting template
    
    dfOut = pd.read_csv(csvName, parse_dates=[0],
                        infer_datetime_format=True, dayfirst=True)
    
    dfOut['Date'] = pd.to_datetime(dfOut['Date'])
    dfOut.set_index('Date', inplace=True)
    dfOut = dfOut.sort_index() # Sort chronologically
    
    return dfOut

#spotNames = dfSpot.columns.get_values()
#dfSpot = importFile('spotEUR.csv')
#dfSpot.head()

## Stat analysis functions ##

def runStats(dfInput, nCol):
    
    dfRets = dfInput.diff()*100 # Daily returns in bps
    dfRets = dfRets.dropna() # First row is NaN, gets dropped, any others too
    
    colNames = dfRets.columns.get_values()
    colNames[nCol]
    
    retRange = dfRets[colNames[nCol]].max()-dfRets[colNames[nCol]].min()

    dfRets[colNames[nCol]].plot(kind='hist',bins=int(retRange),title=colNames[nCol])
    
    return print(colNames[nCol],"\n",
                 'range',retRange,"\n",
                 'mean',np.round(dfRets[colNames[nCol]].mean(),3),"\n",
                 'sigma',np.round(dfRets[colNames[nCol]].std(),3))

## PC Analysis (rolling window) ##

def pcaRun(dfPar):
    
    dfRets = dfPar.diff()*100 # Daily returns in bps
    dfRets = dfRets.dropna() # First row is NaN, gets dropped, any others too
    
    from scipy import stats
    dfRets_o = dfRets[(np.abs(stats.zscore(dfRets)) < 3).all(axis=1)] # Remove 3 sigma outliers
    
    cols = dfRets_o.shape[1]
    pca = PCA(n_components=cols)
    
    eigenVal = []
    eigenVec = []

    sampleSize = dfRets_o.shape[0]
    lookbackSize = 90
    
    for i in range(0,(sampleSize-lookbackSize)):
        #print(i,lookbackSize+i)
        pca.fit(dfRets_o.iloc[i:lookbackSize+i])
        eigenVal.append(pca.explained_variance_ratio_)
        eigenVec.append(pca.components_)
        
    dfPC1 = pd.DataFrame().reindex_like(dfRets_o)
    dfPC1 = dfPC1.iloc[lookbackSize:]
    
    for n in range(0,len(eigenVec)):
        for k in range(0,len(eigenVec[0][0])):
            dfPC1.iloc[n,k] = np.abs(eigenVec[n][0][k])
        
    return dfPC1

def pcaPlot(dfLoad, nDays, firstTenor=0, lastTenor='', mySize=(14,10)):
    
    if lastTenor=='':
        lastTenor = dfLoad.shape[1]
    
    return dfLoad.iloc[dfLoad.shape[0]-nDays:,
                       firstTenor:lastTenor].plot(legend=True,
                                                  figsize=mySize,
                                                  grid=True,
                                                  colormap='rainbow')

## Time series plotting functions ##

def curvePlot(dfInput, nDays, sLong, sShort, mySize=(14,5)):
    
    dfSpread = pd.DataFrame()
    dfSpread = (dfInput[sLong + ' Par Swap Rate']-dfInput[sShort + ' Par Swap Rate'])*100
    
    axSpread = dfSpread.iloc[dfSpread.shape[0]-nDays:].plot(legend=False,
                                                            figsize=mySize,
                                                            title='Curve '+sShort+'/'+sLong,
                                                            grid=True)
    return axSpread

def flyPlot(dfInput, nDays, wing1, belly, wing2, mySize=(14,5)):
    
    dfFly = pd.DataFrame()
    dfFly = ((2*dfInput[belly + ' Par Swap Rate'])
             -dfInput[wing1 + ' Par Swap Rate']
             -dfInput[wing2 + ' Par Swap Rate'])*100
    
    axFly = dfFly.iloc[dfSpread.shape[0]-nDays:].plot(legend=False,
                                                      figsize=mySize,
                                                      title='Fly '+wing1+'/'+belly+'/'+wing2,
                                                      grid=True)
    return axFly

# Column renaming function

def dfRename(dfInput,removeString=' Fwd Swap Rate'):
    
    test = dfInput.columns.get_values()
    
    for x in range(0,len(test)):
        test[x] = test[x].replace(removeString,'')
        
    dfInput.columns = test
    
    return dfInput

### -------------------- EXE -------------------- ###

dfCross = dfRename(csv2df('eurusdFwdnew.csv'),' Basis Fwd Swap Spread')
dfCross = dfCross.dropna()
cols = dfCross.shape[1]

n = 200
m = 200
trainSet = dfCross.iloc[dfCross.shape[0]-n-m:dfCross.shape[0]-n]
testSet = dfCross.iloc[dfCross.shape[0]-n:]

pca = PCA(n_components=3) #n_components=3

pca.fit(trainSet)

inSample = pca.transform(trainSet)
outSample = pca.transform(testSet)

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
print(np.cumsum(pca.explained_variance_ratio_))
plt.show()

d = {'PC1' : pca.components_[0],'PC2' : pca.components_[1],'PC3' : pca.components_[2]} #,'PC3' : pca.components_[2]

#for i in range(0,len(pca.components_)): plt.plot(pca.components_[i])
    
a = {'mats' : dfCross.columns.get_values()}
dfHead = pd.DataFrame(a)
dfPCplot = pd.DataFrame(d)
dfPCplot.plot(figsize=(12,6))
plt.xticks(np.arange(pca.components_.shape[1]),dfHead.mats,rotation=90)
    
plt.show()

#Plotting results of above test

fig = plt.figure(figsize=(12,10)) #figsize=(12,10)
fig.add_axes()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

ax1.plot(inSample)
ax1.set_title('trainSet PCs')
ax2.plot(outSample)
ax2.set_title('testSet PCs')
ax3.plot(pca.inverse_transform(inSample))
ax3.set_title('trainSet Rates')
ax4.plot(pca.inverse_transform(outSample))
ax4.set_title('testSet Rates')

plt.show()

reconOut = pca.inverse_transform(outSample)
resid = testSet - reconOut

resid.plot(legend=True,colormap='rainbow',figsize=(14,10),title='Residuals')
plt.show()
