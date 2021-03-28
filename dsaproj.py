import numpy as np
import matplotlib.pyplot as plt
import lmfit
#import scipy.stats as ss
#import scipy.optimize as so
#import scipy.special as sf


###--------Fitting functions defined----------

def zipf(x, k,s):
	return k*np.float_power(x, -s)
	
def zm(x, k,s,h):
	return k*np.float_power((x+h), -s)

def stretchexp(x, k,h,s):
	return k*np.exp(-np.float_power(h*x, s))
	
def weib(x, k,h,s):
	return k*np.float_power(h*x, s)*np.exp(-np.float_power(h*x, s+1))
	
def parfrac(x, k,s,h):
	return k*np.float_power(x, -s)*np.exp(-h*np.log(x)**2)
	
def cutpow(x, k,s,h):
	return k*np.float_power(x, -s)*np.exp(-h*x)

###unused
'''
def lognorm(x, s,loc,scale):
	return ss.lognorm.pdf(x,s,loc,scale)

def YS(x, a,alpha,loc):
	return a*sf.beta(x-loc, alpha)

def exp(x, k,h):
	return k*np.exp(-h*(x-1))
'''	
	
	
###----------Data reading----------

#www.proteomaps.net
fnames=['mpn_Kuehner.csv', 'eco_Lu.csv', 'syn_Wegener.csv', 'sce_deGodoy.csv', 'spo_Marguerat_proliferating.csv', 'ath_Castellana.csv', 'dme_Brunner_2007.csv', 'hsa_Khan_Chimpanzee_Protein.csv']
names=['M. pneumoniae', 'E. coli', 'Synechocystis sp. 6803', 'S. cerevisiae', 'S. pombe', 'A. thaliana', 'D. melanogaster', 'P. troglodytes']

#this selects the organism/datafile
cur=5

data=np.loadtxt(fnames[cur], skiprows=2, usecols=2) #usecols=2 for abundance (ppm), 4 for sizeweightedabundance(ppm)
data=-np.sort(-data) #sort in descending order
data=data/np.max(data) #normalize
n=len(data)
rank=np.array(range(1,1+n))


###------------Actual fitting------------

#If running the code results in a NaN, try tweaking the initial guesses for the parameters just a little bit. It's probably s that's messing things, or perhaps h.

zipfModel=lmfit.Model(zipf).fit(data, x=rank, k=1.,s=1.)
print('\n', zipfModel.fit_report())

zmModel=lmfit.Model(zm).fit(data, x=rank, k=1.,s=.7,h=1.)
print('\n', zmModel.fit_report())

stretchexpModel=lmfit.Model(stretchexp).fit(data, x=rank, k=1.,h=1.,s=1.)
print('\n', stretchexpModel.fit_report())

params=lmfit.Parameters()
params.add('k', min=0.1)
params.add('h', min=0.0001, max=1.)
params.add('s', min=0.0001, max=1.)
weibModel=lmfit.Model(weib).fit(data, params, x=rank, k=20.,h=0.5,s=0.5)
print('\n', weibModel.fit_report())

parfracModel=lmfit.Model(parfrac).fit(data, x=rank, k=1.,s=0.5,h=0.5)
print('\n', parfracModel.fit_report())

paramsCP=lmfit.Parameters()
paramsCP.add('k', min=0.001)
paramsCP.add('s', min=0.0001)
paramsCP.add('h', min=0.0001, max=1.)
cutpowModel=lmfit.Model(cutpow).fit(data, paramsCP, x=rank, k=1.,s=0.5,h=.05)
print('\n', cutpowModel.fit_report())

#ignore all of these

#expModel=lmfit.Model(exp).fit(data, x=rank, k=1.,h=0.1)
#print('\n', expModel.fit_report())

'''params=lmfit.Parameters()
params.add('s', min=0.1)
params.add('loc', min=0.1)
params.add('scale', min=0.2)
lognormModel=lmfit.Model(lognorm).fit(data, params, x=rank)
print('\n', lognormModel.fit_report())'''
#paramsL=so.curve_fit(lognorm, rank, data)
#print('\n',paramsL)


###------Plotting---------

plt.suptitle(names[cur])
colors=['#1f77b4', '#ff7f0e', '#9cd05c', '#d66768', '#9467bd', '#8c564b', '#e397e2', '#7f7f7f', '#bcbd22', '#17becf']

plt.subplot(1,1,1)
plt.title('frequency-rank (log-log)')
plt.xlabel('protein rank')
plt.ylabel('relative abundance')
plt.xscale('log')
plt.yscale('log')
plt.ylim([0.7*np.min(data), 1.5*np.max(data)])
plt.scatter(rank, data, color=colors[0], label='real data')
plt.plot(rank, zipfModel.best_fit, color=colors[1], label='zipf')
plt.plot(rank, zmModel.best_fit, color=colors[2], label='zipf-mandelbrot')
plt.plot(rank, stretchexpModel.best_fit, color=colors[3], label='stretched exponential')
plt.plot(rank, weibModel.best_fit, color=colors[4], label='weibull')
plt.plot(rank, parfracModel.best_fit, color=colors[5], label='parabolic fractal')
plt.plot(rank, cutpowModel.best_fit, color=colors[6], label='cut-off power-law')
#plt.plot(rank, expModel.best_fit, color=colors[7], label='exponential')
#plt.plot(rank, lognormModel.best_fit, color=colors[5], label='log-normal')
#plt.plot(rank, lognorm(rank, paramsL[0][0],paramsL[0][1]), color=colors[4], label='log-normal')
plt.legend()

#residuals
'''plt.subplot(1,2,2)
plt.hist(zmModel.best_fit-data, bins=int(1+3.3*np.log(n)), color=colors[2], alpha=0.2, density=True, label='zipf-mandelbrot')
plt.hist(stretchexpModel.best_fit-data, bins=int(1+3.3*np.log(n)), color=colors[3], alpha=0.2, density=True, label='stretched exponential')
plt.hist(parfracModel.best_fit-data, bins=int(1+3.3*np.log(n)), color=colors[5], alpha=0.2, density=True, label='parabolic fractal')
plt.hist(cutpowModel.best_fit-data, bins=int(1+3.3*np.log(n)), color=colors[6], alpha=0.2, density=True, label='cut-off power-law')
'''

plt.show()