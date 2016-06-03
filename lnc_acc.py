#Python 2.7
#Written by Shuyang Gao (BiLL), email: gaos@usc.edu


from scipy import stats
import numpy as np
import scipy.spatial as ss
from scipy.special import digamma,gamma
from sklearn.neighbors import KDTree, DistanceMetric
import numpy.random as nr
import random
import matplotlib.pyplot as plt
import re
from scipy.stats.stats import pearsonr
import numpy.linalg as la
from numpy.linalg import eig, inv, norm, det
from scipy import stats
from math import log,pi,hypot,fabs,sqrt
class MI:

    @staticmethod
    def zip2(*args):
        #zip2(x,y) takes the lists of vectors and makes it a list of vectors in a joint space
        #E.g. zip2([[1],[2],[3]],[[4],[5],[6]]) = [[1,4],[2,5],[3,6]]
        return [sum(sublist,[]) for sublist in zip(*args)]

    @staticmethod
    def avgdigamma(points,dvec,metric='minkowski', p=float('inf')):
        tree = KDTree(points, metric=DistanceMetric.get_metric(metric,p=p))
        num_points = tree.query_radius(points, dvec - 1e-15, count_only=True)
        return np.sum(digamma(num_points) / len(points) )

    @staticmethod
    def mi_Kraskov(x,y,k=5,base=np.exp(1),intens=1e-10,metric="minkowski",p=np.float64('inf')):
        '''The mutual information estimator by Kraskov et al.
           Inputs are 2D arrays, with each column being a dimension and each row being a data point
        '''
        assert len(x)==len(y), "Lists should have same length"
        assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
        x +=  intens*nr.rand(x.shape[0],x.shape[1])
        y +=  intens*nr.rand(x.shape[0],x.shape[1])
        points = np.hstack((x,y))

        #Find nearest neighbors in joint space, p=inf means max-norm
        tree = KDTree(points, metric=DistanceMetric.get_metric(metric,p=p))
        try:
          dvec = tree.query(points,k+1)[0][:,k]   # no need to reshape with new query_radius method
        except ValueError:
          return (float("NaN"))

        a = MI.avgdigamma(x,dvec*x.shape[1]/points.shape[1],metric=metric,p=p)
        b = MI.avgdigamma(y,dvec*y.shape[1]/points.shape[1],metric=metric,p=p)
        c = digamma(k)
        d = digamma(len(x))
        # print("ee_acc: %s, %s, %s, %s" %( a,b,c,d))
        return (-a-b+c+d)/np.log(base)

    @staticmethod
    def mi_LNC(x,y,k=5,base=np.exp(1),alpha=0.25,intens = 1e-10,metric='minkowski',p=np.float64('inf')):
        '''The mutual information estimator by PCA-based local non-uniform correction(LNC)
           ith row of X represents ith dimension of the data, e.g. X = [[1.0,3.0,3.0],[0.1,1.2,5.4]], if X has two dimensions and we have three samples
           alpha is a threshold parameter related to k and d(dimensionality), please refer to our paper for details about this parameter
        '''
        #N is the number of samples
        N = x.shape[0]

        #First Step: calculate the mutual information using the Kraskov mutual information estimator
        #adding small noise to X, e.g., x<-X+noise
        x += intens*nr.rand(x.shape[0],x.shape[1])
        y += intens*nr.rand(x.shape[0],x.shape[1])
        points = np.hstack((x,y))

        tree = KDTree(points, metric=DistanceMetric.get_metric(metric, p=p))
        try:
            dvec, knn_idx = tree.query(points, k+1)   # no need to reshape with new query_radius method
        except ValueError:
          return (float("NaN"))

        a = MI.avgdigamma(x,dvec[:,-1]*x.shape[1]/points.shape[1], metric=metric, p=p)
        b = MI.avgdigamma(y,dvec[:,-1]*y.shape[1]/points.shape[1], metric=metric, p=p)
        c = digamma(k)
        d = digamma(len(x))

        # a,b,c,d = MI.avgdigamma(x,dvec), MI.avgdigamma(y,dvec), digamma(k), digamma(len(x))
        # print("ee_acc: %s, %s, %s, %s" %( a,b,c,d))
        ret = (-a-b+c+d)/np.log(base)

        # LNC correction
        logV_knn = np.sum(np.log(np.abs(points - points[knn_idx[:,-1],:])), axis=1)
        logV_projected = np.zeros(logV_knn.shape)
        for i in range(points.shape[0]):
            knn_points = points[knn_idx[i,:],:]
            knn_centered = knn_points - points[i,:]
            u,s,v = la.svd(knn_centered)
            knn_proj = knn_centered.dot(v.T)
            max_dims = np.max(np.abs(knn_proj), axis=0)   # max-norm per dimension
            logV_projected[i] = np.sum(np.log(max_dims))

        diff = logV_projected - logV_knn
        if (alpha>1): alpha = 1
        diff[diff >= log(alpha)] = 0
        e = -np.sum(diff) / N

        return (ret + e)/log(base);

    @staticmethod
    def entropy(x,k=3,base=np.exp(1),intens=1e-10):
        """ The classic K-L k-nearest neighbor continuous entropy estimator
            x should be a list of vectors, e.g. x = [[1.3],[3.7],[5.1],[2.4]]
            if x is a one-dimensional scalar and we have four samples
        """
        assert k <= len(x)-1, "Set k smaller than num. samples - 1"
        d = len(x[0])
        N = len(x)
        x +=  intens*nr.rand(N,d)
        tree = KDTree(x, metric=DistanceMetric.get_metric("minkowski",p=np.float64('inf') ))
        nn = tree.query(x,k+1)[0][:,k]   # no need to reshape with new query_radius method
        const = digamma(N)-digamma(k) + d*log(2)
        return (const + d*np.mean(map(log,nn)))/log(base)
