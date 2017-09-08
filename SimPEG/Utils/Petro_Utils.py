import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.mixture.gaussian_mixture import _compute_precision_cholesky
from .matutils import mkvc


def ComputeDistances(a,b):

    x = mkvc(a, numDims=2)
    y = mkvc(b, numDims=2)

    n, d = x.shape
    t, d1 = y.shape

    assert d == d1, ('vectors must have same number of columns')

    sq_dis = np.dot((x**2.), np.ones([d, t]))+np.dot(np.ones([n, d]), (y**2.).T)-2.*np.dot(x, y.T)

    idx = np.argmin(sq_dis, axis=1)

    return sq_dis**0.5, idx


def order_clusters_GM_weight(GMmodel, outputindex=False):
    '''
    order cluster by increasing mean for Gaussian Mixture scikit object
    '''

    indx = np.argsort(GMmodel.weights_, axis=0)[::-1]
    GMmodel.means_ = GMmodel.means_[indx].reshape(GMmodel.means_.shape)
    GMmodel.weights_ = GMmodel.weights_[indx].reshape(GMmodel.weights_.shape)
    if GMmodel.covariance_type == 'tied':
        pass
    else:
        GMmodel.precisions_ = GMmodel.precisions_[indx].reshape(GMmodel.precisions_.shape)
        GMmodel.covariances_ = GMmodel.covariances_[indx].reshape(GMmodel.covariances_.shape)
    GMmodel.precisions_cholesky_ = _compute_precision_cholesky(GMmodel.covariances_, GMmodel.covariance_type)

    if outputindex:
        return indx


def order_cluster(GMmodel,GMref,outputindex = False):
    order_clusters_GM_weight(GMmodel)

    idx_ref = np.ones_like(GMref.means_,dtype=bool)

    indx = []

    for i in range(GMmodel.n_components):
        _, id_dis = ComputeDistances(mkvc(GMmodel.means_[i],numDims=2),
                                     mkvc(GMref.means_[idx_ref],numDims=2))
        idrefmean = np.where(GMref.means_ == GMref.means_[idx_ref][id_dis])[0][0]
        indx.append(idrefmean)
        idx_ref[idrefmean] = False

    GMmodel.means_ = GMmodel.means_[indx].reshape(GMmodel.means_.shape)
    GMmodel.weights_ = GMmodel.weights_[indx].reshape(GMmodel.weights_.shape)
    if GMmodel.covariance_type == 'tied':
        pass
    else:
        GMmodel.precisions_ = GMmodel.precisions_[indx].reshape(GMmodel.precisions_.shape)
        GMmodel.covariances_ = GMmodel.covariances_[indx].reshape(GMmodel.covariances_.shape)
    GMmodel.precisions_cholesky_ = _compute_precision_cholesky(GMmodel.covariances_, GMmodel.covariance_type)

    if outputindex:
        return indx


def computePrecision(GMmodel):
    if GMmodel.covariance_type == 'full':
        GMmodel.precisions_ = np.empty(GMmodel.precisions_cholesky_.shape)
        for k, prec_chol in enumerate(GMmodel.precisions_cholesky_):
            GMmodel.precisions_[k] = np.dot(prec_chol, prec_chol.T)

    elif GMmodel.covariance_type == 'tied':
        GMmodel.precisions_ = np.dot(GMmodel.precisions_cholesky_,
                                      GMmodel.precisions_cholesky_.T)
    else:
        GMmodel.precisions_ = GMmodel.precisions_cholesky_ ** 2

def computeCovariance(GMmodel):
    if GMmodel.covariance_type == 'full':
        GMmodel.covariances_ = np.empty(GMmodel.covariances_cholesky_.shape)
        for k, cov_chol in enumerate(GMmodel.covariances_cholesky_):
            GMmodel.covariances_[k] = np.dot(cov_chol, cov_chol.T)

    elif GMmodel.covariance_type == 'tied':
        GMmodel.covariances_ = np.dot(GMmodel.covariances_cholesky_,
                                      GMmodel.covariances_cholesky_.T)
    else:
        GMmodel.covariances_ = GMmodel.covariances_cholesky_ ** 2
