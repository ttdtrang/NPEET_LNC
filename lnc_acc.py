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
    def avgdigamma(points,dvec,metric='minkowski',p=float('inf')):
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

        a = avgdigamma(x,dvec*x.shape[1]/points.shape[1],metric=metric,p=p)
        b = avgdigamma(y,dvec*y.shape[1]/points.shape[1],metric=metric,p=p)
        c = digamma(k)
        d = digamma(len(x))
        # print("ee_acc: %s, %s, %s, %s" %( a,b,c,d))
        return (-a-b+c+d)/np.log(base)

    @staticmethod
    def mi_LNC(x,y,k=5,base=np.exp(1),alpha=0.25,intens = 1e-10):
        '''The mutual information estimator by PCA-based local non-uniform correction(LNC)
           ith row of X represents ith dimension of the data, e.g. X = [[1.0,3.0,3.0],[0.1,1.2,5.4]], if X has two dimensions and we have three samples
           alpha is a threshold parameter related to k and d(dimensionality), please refer to our paper for details about this parameter
        '''
        #N is the number of samples
        N = len(X[0]);

        #First Step: calculate the mutual information using the Kraskov mutual information estimator
        #adding small noise to X, e.g., x<-X+noise
        x +=  intens*nr.rand(x.shape[0],x.shape[1])
        y +=  intens*nr.rand(x.shape[0],x.shape[1])
        points = np.hstack((x,y))

        tree = KDTree(points, metric=DistanceMetric.get_metric("minkowski",p=np.float64('inf') ))
        try:
          dvec = tree.query(points,k+1)[0][:,k]   # no need to reshape with new query_radius method
        except ValueError:
          return (float("NaN"))

        a,b,c,d = MI.avgdigamma(x,dvec), MI.avgdigamma(y,dvec), digamma(k), digamma(len(x))
        # print("ee_acc: %s, %s, %s, %s" %( a,b,c,d))
        ret = (-a-b+c+d)/np.log(base)

        #Second Step: Add the correction term (Local Non-Uniform Correction)
        e = 0.
        tot = -1;
          for point in points:
            tot += 1;
            #Find k-nearest neighbors in joint space, p=inf means max norm
            knn_points = points[tree.query(points,k+1,p=float('inf'))[1],:]
            knn_points = knn_points - knn_points[0,:]
            covr = np.outer(knn_points[1:,:].T, knn_points[1:,:]) / k
            w,v = la.eig(covr)
            #Substract mean of k-nearest neighbor points (For each dimension, shift every points such as the center (0-distance) is at 0
            # for i in range(len(point)):
            #     avg = knn_points[0][i];
            #     for j in range(k+1):
            #         knn_points[j][i] -= avg;

            #Calculate covariance matrix of k-nearest neighbor points, obtain eigen vectors
            # covr = [];
            # for i in range(len(point)):
            #     tem = 0;
            #     covr.append([]);
            #     for j in range(len(point)):
            #         covr[i].append(0);
            # for i in range(len(point)):
            #     for j in range(len(point)):
            #         avg = 0.
            #         for ii in range(1,k+1):
            #             avg += knn_points[ii][i] * knn_points[ii][j] / float(k);
            #         covr[i][j] = avg;
            # w, v = la.eig(covr);


            #Calculate PCA-bounding box using eigen vectors
            # V_rect = 0;
            # cur = [];
            # for i in range(len(point)):
            #     maxV = 0.
            #     for j in range(0,k+1):
            #         tem = 0.;
            #         for jj in range(len(point)):
            #             tem += v[jj,i] * knn_points[j][jj];
            #         if fabs(tem) > maxV:
            #             maxV = fabs(tem);
            #     cur.append(maxV);
            #     V_rect = V_rect + log(cur[i]);

            #Calculate the volume of original box
            log_knn_dist = 0.;
            for i in range(len(dvec)):
                log_knn_dist += log(dvec[i][tot]);

            #Perform local non-uniformity checking
            if V_rect >= log_knn_dist + log(alpha):
                V_rect = log_knn_dist;

            #Update correction term
            if (log_knn_dist - V_rect) > 0:
                e += (log_knn_dist - V_rect)/N;

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
