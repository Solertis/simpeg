from SimPEG import Mesh, Problem, Survey, Maps, Utils, EM, DataMisfit, Regularization, Optimization, InvProblem, Directives, Inversion
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
from pymatsolver import PardisoSolver
from scipy.stats import norm,multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.mixture.gaussian_mixture import _compute_precision_cholesky

np.random.seed(1)
N = 100
mesh = Mesh.TensorMesh([N])

nk = 20
jk = np.linspace(1., 60., nk)
p = -0.25
q = 0.25


def g(k):
    return (
        np.exp(p*jk[k]*mesh.vectorCCx) *
        np.cos(np.pi*q*jk[k]*mesh.vectorCCx)
        )

G = np.empty((nk, mesh.nC))

for i in range(nk):
    G[i, :] = g(i)

mtrue = np.zeros(mesh.nC)
mtrue[mesh.vectorCCx > 0.3] = 1.
mtrue[mesh.vectorCCx > 0.45] = -0.5
mtrue[mesh.vectorCCx > 0.6] = 0

prob = Problem.LinearProblem(mesh, G=G)
survey = Survey.LinearSurvey()
survey.pair(prob)
survey.makeSyntheticData(mtrue, std=0.01)

M = prob.mesh

reg = Regularization.Tikhonov(mesh, alpha_s=1., alpha_x=1.)
dmis = DataMisfit.l2_DataMisfit(survey)
opt = Optimization.InexactGaussNewton(maxIter=60)
invProb = InvProblem.BaseInvProblem(dmis, reg, opt)
directives = [
    Directives.BetaEstimate_ByEig(beta0_ratio=1e-2),
    Directives.TargetMisfit(),
    #PlotIter()
    ]

inv = Inversion.BaseInversion(invProb, directiveList=directives)
m0 = np.zeros_like(survey.mtrue)

mrec = inv.run(m0)

# fit a Gaussian Mixture Model with n components
n = 3
clfnormal = GaussianMixture(n_components=n, covariance_type='full',max_iter=1000, n_init=20)
clfnormal.fit(mrec.reshape(-1,1))

clf = GaussianMixture(n_components=n, covariance_type='full',max_iter=1000, n_init=20)
clf.fit(mtrue.reshape(-1,1))

mnormal = mrec

# Petrophysics inversion
regmesh = mesh;
m0 = np.median(clf.means_)*np.ones(mesh.nC);
dmis = DataMisfit.l2_DataMisfit(survey)
#reg = Regularization.Tikhonov(regmesh)#,mapping = mapping)#,indActive=actind)
reg = Regularization.PetroRegularization(GMmodel = clf, mesh = regmesh)#, mapping = mapping, indActive=actind)

minit = m0

reg.mref = minit
reg.membership = clf.predict(minit.reshape(-1,1))
reg.mrefInSmooth = True
reg.alpha_s = 0.
reg.alpha_x = 1.
#reg.alpha_y = 0.
#reg.alpha_z = 0.
reg.alpha_nll= 1.
opt = Optimization.InexactGaussNewton(maxIter=40,tolX=1e-6)
opt.remember('xc')
invProb = InvProblem.BaseInvProblem(dmis, reg, opt)
beta = Directives.BetaEstimate_ByEig(beta0_ratio=1e-3)

betapetro = np.ones(clf.n_components)*.75
invProb.betapetro = betapetro
invProb.petromodelRef = clf

invProb.reg.GMmodel = clf

petrodir = Directives.GaussianMixtureUpdateModel(coolingFactor = 1., coolingRate = 100, update_covariances = False, verbose = False)

inv = Inversion.BaseInversion(invProb, directiveList=[beta, petrodir]) #beta,Plot

mcluster= inv.run(minit)

fig, axes = plt.subplots(1, 2, figsize=(12*1.2, 4*1.2))
for i in range(prob.G.shape[0]):
    axes[0].plot(prob.G[i, :])
axes[0].set_title('Columns of matrix G')

axes[1].plot(M.vectorCCx, survey.mtrue, color='black')
axes[1].plot(M.vectorCCx, mrec, color='blue')
#axes[1].plot(M.vectorCCx, clf.predict(mcluster.reshape(-1,1)), 'r-')
axes[1].plot(M.vectorCCx, mcluster, 'r-')
axes[1].plot(M.vectorCCx,invProb.reg.mref, 'r--')

axes[1].legend(('True Model', 'L2 Model', 'Petro Model','Learned Mref'))
axes[1].set_ylim([-2, 2])

fig0 = plt.figure(figsize=(12,4))

clfref = invProb.petromodelRef
clfinv = invProb.reg.GMmodel
testXplot = np.linspace(-2.,2.,1000)[:,np.newaxis];
log_dens0 = clfref.score_samples(testXplot);
log_dens = clfinv.score_samples(testXplot);

ax0 = fig0.add_subplot(131)
ax1 = fig0.add_subplot(132)
ax2 = fig0.add_subplot(133)
ax0.plot(testXplot, np.exp(log_dens0),linewidth =3.,color='black')
ax0.plot(testXplot, np.exp(log_dens),linewidth =2.,color='orange')
ax0.set_xlabel('ln_conductivity')
ax0.hist(invProb.model,normed = True, bins = 100);
ax0.set_ylim([0.,5.])

ax1.plot(M.vectorCCx, invProb.reg.GMmodel.means_[invProb.reg.GMmodel.predict(invProb.model.reshape(-1,1))], color='black')
ax1.plot(M.vectorCCx, invProb.model, color='red')

ax2.plot(testXplot,invProb.reg.GMmodel.predict(testXplot),color='red')
ax2.plot(testXplot,invProb.petromodelRef.predict(testXplot),color='black')

plt.show()
