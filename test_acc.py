# Python 2.7
# Testing local non-uniform estimators

from lnc_acc import MI
import entropy_estimators_acc as ee
from math import log,pi
import numpy as np
import numpy.random as nr
import random


nr.seed(471776)
N = 5000; #total number of samples
usedN = 100; #number of samples used for calculation

#2D Linear
noise = 1e-7
intens = 1e-10
x = nr.random((N,1))
y = x + nr.random((N,1))*noise

print 'Testing 2D linear relationship Y=X+Uniform_Noise'
print 'noise level=' + str(noise) + ", Nsamples = " + str(usedN);
print 'True MI(x:y)', MI.entropy(y[:1000],k=3,base=np.exp(1),intens=intens) - log(noise);
print 'Kraskov MI(x:y)', MI.mi_Kraskov(x[:usedN,],y[:usedN],k=3,base=np.exp(1),intens=intens);
print 'Kraskov MI(x:y) (GvS implemetation)', ee.mi(x[:usedN,],y[:usedN],k=3,base=np.exp(1),intens=intens);
print 'LNC MI(x:y)'
for alpha in [0.4, 0.3, 0.25, 0.1, 1e-2, 1e-5, 1e-6, 1e-7]:
    print "alpha = %.1e, MI = %5.3f" % (alpha, MI.mi_LNC(x[:usedN,],y[:usedN,],k=3,base=np.exp(1),alpha=alpha,intens=intens))

# 2D Quadratic
y = x**2 +nr.random((N,1))*noise
usedN = 100; #number of samples used for calculation
print 'Testing 2D quadratic relationship Y=X^2+Uniform_Noise'
print 'noise level=' + str(noise) + ", Nsamples = " + str(usedN);
print 'True MI(x:y)', MI.entropy(y[:1000],k=3,base=np.exp(1),intens=intens) - log(noise);
print 'Kraskov MI(x:y)', MI.mi_Kraskov(x[:usedN,],y[:usedN],k=3,base=np.exp(1),intens=intens);
print 'Kraskov MI(x:y) (GvS implemetation)', ee.mi(x[:usedN,],y[:usedN],k=3,base=np.exp(1),intens=intens);
print 'LNC MI(x:y)'
for alpha in [0.4, 0.3, 0.25, 0.1, 1e-2, 1e-5, 1e-6, 1e-7]:
    print "alpha = %.1e, MI = %5.3f" % (alpha, MI.mi_LNC(x[:usedN,],y[:usedN,],k=3,base=np.exp(1),alpha=alpha,intens=intens))


