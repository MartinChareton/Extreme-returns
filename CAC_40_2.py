# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 11:47:57 2020

@author: Martin Chareton
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
#import datetime as dt
#from statsmodels.tsa.stattools import adfuller
import powerlaw
from arch import arch_model
import scipy.stats

#%% Importing and preparing data
#import CAC 40 daily indices historic from Euronext
dfc = pd.read_csv('C:/Users/Martin Chareton/Documents/Sorbonne/QMF/Output and Stock prices Distributions/CAC 40_quote_chart.csv')

dfc = dfc.drop(["volume;"],1)

dfc.plot()

dfc['rCAC'] = dfc['CAC 40'] / dfc['CAC 40'].shift(1) - 1

dfc.dropna(how="any", inplace=True)

plt.plot(dfc['rCAC'])

plt.hist(dfc['rCAC'], bins = 50)

#Performing Augmented Dickey Fuller test on returns
#import frpl
from frpl import adf_test
adf_test(dfc['rCAC']) # singificance level of 5% by default

#Plot an autocorrelation function for independence visual inspection
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(dfc['rCAC'])
fig = plot_acf(dfc['rCAC'])
#Ljung-Box tests : are the returns independent ?
import statsmodels.api as sm
sm.stats.acorr_ljungbox(dfc['rCAC'], lags=[3], return_df=True)

#Draw QQplot to get an idea of normality
from statsmodels.graphics.gofplots import qqplot
qqplot(dfc['rCAC'], line='s')
plt.title('CAC 40 returns Q-Q plot')
plt.savefig("C:/Users/Martin Chareton/Documents/Sorbonne/QMF/Output and Stock prices Distributions/Graphiques/CAC40_QQplot")

#Normal distribution Shapiro-Wilk test
from scipy.stats import shapiro
stat, p = shapiro(dfc['rCAC'])
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('Sample looks Gaussian (fail to reject H0)')
else:
	print('Sample does not look Gaussian (reject H0)')

#%% PowerLaw fit
#create a class to settle fit, xmin, alpha, for a given set of data and its tails
class data_fit:
    def __init__(self, dataframe, column, thresholdpos, thresholdneg):
        self.postail = dataframe.loc[dataframe[column]>thresholdpos,column]
        self.negtail = -dataframe.loc[dataframe[column]<-thresholdneg,column]
        self.xminpos = 0
        self.xminneg = 0
        self.alphapos = 0
        self.alphaneg = 0
    def fit(self):
        self.fit = powerlaw.Fit(self.postail)
        self.xminpos = self.fit.power_law.xmin
        self.alphapos = self.fit.power_law.alpha
        print('postail')
        print('alpha ',self.fit.power_law.alpha)
        print('x_min ',self.fit.power_law.xmin)
        print('sigma ',self.fit.power_law.sigma)
        
        self.fit = powerlaw.Fit(self.negtail)
        print('negtail')
        print('alpha ',self.fit.power_law.alpha)
        print('x_min ',self.fit.power_law.xmin)
        print('sigma ',self.fit.power_law.sigma)
        self.xminneg = self.fit.power_law.xmin
        self.alphaneg = self.fit.power_law.alpha
        #self.compare = powerlaw.Fit(self.postail).distribution_compare('power_law', 'exponential', normalized_ratio = True)

cac_40 = data_fit(dfc, 'rCAC', 0.0125, 0.0125)
cac_40.fit()

#define a function to plot a powerlaw
def powerfunc(x, alpha, xmin):
    return (alpha-1)*xmin**(alpha-1) * x**(-alpha)

xmin = cac_40.xminpos
xmax = 0.06
xrange = np.arange(xmin, xmax, 0.002)
alpha = cac_40.alphapos

#construit un dataframe avec les couples (x,y) de la fonction puissance
dfplot = pd.DataFrame(index=xrange,columns=['Powerlaw_fit'])
dfplot['Powerlaw_fit'] = powerfunc(dfplot.index, alpha, xmin)

#Superposer la distribution empirique avec le modèle calibré sur la queue des valeurs postitives
ax = dfplot.plot(title='Model fitting to empirical CAC 40 positive tail', color = "r")
ax.set_xlabel('x')
ax.set_ylabel('p(x)')
fig = ax.get_figure()
plt.hist(dfc['rCAC'], bins = 100, label='empirical returns', alpha=0.5)
plt.hist(vrs, bins = 100, label='experimental powerlaw distribution', alpha=0.8)
#Réduit le champ du graphique aux valeurs positives
plt.xlim(0,xmax)
#put a legend for the labels to appear
plt.legend(loc='best')
plt.show()
fig.savefig('C:/Users/Martin Chareton/Documents/Sorbonne/QMF/Output and Stock prices Distributions/Graphiques/fit_to_CAC40.png')


#%%Kolmogorow smirnoff test to assess goodness of fit

#Create a dataframe with all the fitted returns
dfneg_cac = dfc[dfc['rCAC']<-cac_40.xminneg]*(-1)
dfpos_cac = dfc[dfc['rCAC']>cac_40.xminpos]
#set N equals the number of fitted values either positive or negative
N = len(dfpos_cac)
#generates a power law distribution with the fit parameters
vrs = powerlaw.Power_Law(xmin=xmin, parameters=[cac_40.alphapos]).generate_random(N)
dfpos['Random_pl'] = vrs
#visualizes the random draw from the theoretical Powerlaw
plt.hist(vrs, bins = 50, label='empirical returns', alpha=0.5)
plt.xlim(0,0.10)
#perform the kolmogrof smirnoff test to compare emprical and random draw
scipy.stats.ks_2samp(dfpos.rCAC, dfpos.Random_pl)

#create a function to perform X goodness of fit KS tests
def goodness(data_fit, df, column, testnb):
    data_fit.fit
    xmin = data_fit.xminpos
    #Create a dataframe with all the fitted returns
    dfpos = df[df[column]>xmin]
    N = len(dfpos)
    #generates X random vectors of variables of power law distribution with the fit parameters and apply KS
    X = testnb
    KS = 0
    test = []
    check = []
    for i in range(0,X):
        vrs = powerlaw.Power_Law(xmin=xmin, parameters=[data_fit.alphapos]).generate_random(N)
        dfpos['Random_pl'] = vrs
        KS = scipy.stats.ks_2samp(dfpos[column], dfpos.Random_pl)[1]
        test.append(KS)
        if KS > 0.10:
            check.append(1)
        else:
            check.append(0)
    #we check the percentage of time that we failed to reject H0
    print('fail to reject rate', mean(check))
    #we compute the average KS p value, i.e likelihood of the data being drawn from the same samples given the random and empirical drawns
    print('p-value mean', mean(test))
    
goodness(cac_40, dfc, 'rCAC', 2500)
    
#create a function to perform goodness of fit on left tail and takes in parameters the dataframe
def goodness_neg(data_fit, df, column, testnb):
    data_fit.fit
    xmin = data_fit.xminneg
    #Create a dataframe with all the fitted returns
    dfpos = df[df[column]<-xmin]*(-1)
    N = len(dfpos)
    #generates X random vectors of variables of power law distribution with the fit parameters and apply KS
    X = testnb
    KS = 0
    test = []
    check = []
    for i in range(0,X):
        vrs = powerlaw.Power_Law(xmin=xmin, parameters=[data_fit.alphaneg]).generate_random(N)
        dfpos['Random_pl'] = vrs
        KS = scipy.stats.ks_2samp(dfpos[column], dfpos.Random_pl)[1]
        test.append(KS)
        if KS > 0.10:
            check.append(1)
        else:
            check.append(0)
    #we check the percentage of time that we failed to reject H0
    print('fail to reject rate', mean(check))
    #we compute the average KS p value, i.e likelihood of the data being drawn from the same samples given the random and empirical drawns
    print('p-value mean', mean(test))


cac_40.xminneg
cac_40.alphaneg
#Create a dataframe with all the fitted returns
dfpos_cac = dfc[dfc['rCAC']<-cac_40.xminneg]*(-1)
N = len(dfpos_cac)
#generates a power law distribution with the fit parameters
vrs = powerlaw.Power_Law(xmin=cac_40.xminneg, parameters=[cac_40.alphaneg]).generate_random(N)
dfpos_cac['Random_pl'] = vrs
#visualizes the random draw from the theoretical Powerlaw
plt.hist(vrs, bins = 50, label='empirical returns', alpha=0.5)
plt.xlim(0,0.10)
#perform the kolmogrof smirnoff test to compare emprical and random draw
scipy.stats.ks_2samp(dfpos_cac.rCAC, dfpos_cac.Random_pl)


goodness_neg(cac_40, dfc, 'rCAC', 2500)


#%%GARCH filtering
#Fit a GARCH model on returns
garch11 = arch_model(dfc.rCAC, p=1, q=1)
res = garch11.fit()
print(res.summary())
res.plot()
res.plot().savefig('C:/Users/Martin Chareton/Documents/Sorbonne/QMF/Output and Stock prices Distributions/Graphiques/GARCH_StandardizedResiduals.png')
res.conditional_volatility.plot()
dir(res)
res.resid.plot()
dfc.rCAC.plot()
# both look the same, we plot the standardized residuals
res.std_resid.plot()
dfc.rCAC.plot()

#Test standardized redisuals autocorrelation
#Plot an autocorrelation function for visual inspection
plot_acf(res.std_resid)
fig = plot_acf(res.std_resid)
fig.savefig('C:/Users/Martin Chareton/Documents/Sorbonne/QMF/Output and Stock prices Distributions/Graphiques/GARCH_Residuals_ACF.png')

#Ljung-Box tests : are the residuals pure innovations or are they serially correlated ?
sm.stats.acorr_ljungbox(res.std_resid, lags=[10], return_df=True)



#%%Test the normality of the GARCH fit innovations

#Visual inspection
dfgarch = pd.concat([dfc.rCAC,res.std_resid], axis=1)
dfgarch.columns = ['CAC 40 returns','Garch(1,1) Standardized Residuals']
dfgarch['Innovations_Normal'] = np.random.normal(dfgarch['Garch(1,1) Standardized Residuals'].mean(), dfgarch['Garch(1,1) Standardized Residuals'].std(), len(dfgarch))
plt.hist(dfgarch['Garch(1,1) Standardized Residuals'], 100, alpha=0.5 ,density=True, label='GARCH Innovations')
plt.hist(dfgarch.Innovations_Normal, 100, alpha=0.5 ,density=True, label='Residuals if normal')
plt.legend(loc='upper right')
plt.title('GARCH Residuals of CAC returns versus normal law')
plt.xlim(-4,4)
#plt.savefig('C:/Users/Martin Chareton/Documents/Sorbonne/QMF/Output and Stock prices Distributions/Graphiques/GARCH_innovations_hist.png')
plt.show()

#Jarque-Bera test
from scipy.stats import kurtosis
from scipy.stats import skew
JBstat = (len(dfgarch) / 6) * skew(dfgarch['Garch(1,1) Standardized Residuals'])**2 + (len(dfgarch) / 24 ) * (kurtosis(dfgarch['Garch(1,1) Standardized Residuals']) - 3)**2
print(JBstat)
# compare it to a chi 2 at 95% with two degrees of freedom
scipy.stats.chi2.ppf(q = 0.95, df = 2)

#Import directly the value of Kurtosis and skewness
import statsmodels.stats.api as sms
from statsmodels.compat import lzip
name = ['Jarque-Bera test', 'Chi-squared(2) p-value', 'Skewness', 'Kurtosis']
test = sms.jarque_bera(dfgarch['Garch(1,1) Standardized Residuals'])
lzip(name, test)

#Shapiro-Wilk normality test
from scipy.stats import shapiro
stat, p = shapiro(dfgarch['Garch(1,1) Standardized Residuals'])
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('Sample looks Gaussian (fail to reject H0)')
else:
	print('Sample does not look Gaussian (reject H0)')
    
scipy.stats.ks_2samp(dfgarch['Garch(1,1) Standardized Residuals'], dfgarch['Innovations_Normal'])

#Searching for a power law on the residuals
#defining a class of the std redisuals of the CAC_40 returns
cac_40_resid = data_fit(dfgarch,'Garch(1,1) Standardized Residuals', 1, 1)
#fit a power law
cac_40_resid.fit()
#assess goodness of fit
goodness(cac_40_resid, dfgarch, 'Garch(1,1) Standardized Residuals', 2500)

goodness_neg(cac_40_resid, dfgarch, 'Garch(1,1) Standardized Residuals', 2500)

dfpos_cac_in = dfgarch[dfgarch['Garch(1,1) Standardized Residuals']>cac_40_resid.xminpos]
dfneg_cac_in = dfgarch[dfgarch['Garch(1,1) Standardized Residuals']<-cac_40_resid.xminneg]*(-1)
N = len(dfpos_cac_in)


#%% GEV Generalized Extreme Value - Inspired from An Application of Extreme Value Theory for Measuring Financial Risk by Gilli & Kellezi
#implemented by Lina Jouker

from scipy.stats import genextreme as gev
from scipy.optimize import curve_fit

# Method of Block Maxima: 

# Step 1 - Divide the sample in n blocks of equal length

print(dfc['rCAC'].shape) # Initial shape of our data 

list_of_df = np.array_split(dfc['rCAC'], 45) # Spliting it to 45 equal parts, like Gilli & Kellezi 

print(list_of_df[1].shape) # Shape of second chunk as an example

max_values = [] # Initiliazing the array that will hold our maxima

# Step 2 - Collecting the maximum value in each block

for i in range (0, len(list_of_df) - 1):
	max_values.append(max(list_of_df[i]))

print(max_values)

# Find the set of Maxima by their index:

# Find the maximum of maximums

maxima = max(max_values)
print(maxima)

# Find the index of the set it belongs to

maxima_index = max_values.index(maxima)
print(maxima_index)

# Save set of maxima
set_of_maxima = list_of_df[maxima_index]
  
# Step 3 - Fitting the GEV distribution to the set of maxima

def main(max_values): # We define a function that specifies the 3 parameters of GEV
    shape, loc, scale = gev.fit(max_values)
    return shape, loc, scale

shape, loc, scale = main(max_values)

print(shape)
print(loc)
print(scale)

l = loc + scale / shape
x = np.linspace(0.017, 0.11, num=44)
y = gev.pdf(x, shape, loc, scale) # We plot our GEV cdf

hist, bins = np.histogram(max_values, bins=10, range=(0.017, 0.11), density=True) # We plot our CAC 40 return maxima histogram
plt.bar(bins[:-1], hist, width = 0.01, align='edge')
plt.plot(x, y, 'ro')
plt.title("GEV fit on maxima") 
plt.show() 


