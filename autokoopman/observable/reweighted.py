from autokoopman.observable import SymbolicObservable, KoopmanObservable
import math
import numpy as np
import sympy as sp
from scipy.stats import cauchy, laplace
from scipy.optimize import nnls


def make_gaussian_kernel(sigma=1.0):
  """the baseline"""
  def kernel(x, xp):
    return np.exp(-np.linalg.norm(x-xp)**2.0 / (2.0*sigma**2.0))
  return kernel


def rff_reweighted_map(n, X, Y, wx, wy, sigma=1.0):
  """build a RFF explicit feature map"""
  assert len(X) == len(Y)
  R = n
  D = X.shape[1]
  N = len(X)
  W    = np.random.normal(loc=0, scale=(1.0/np.sqrt(sigma)), size=(R, D))
  b    = np.random.uniform(-np.pi, np.pi, size = R)

  # get ground truth kernel for training
  kernel = make_gaussian_kernel(sigma)

  # solve NNLS problem
  M = np.zeros((N, R))
  bo = np.zeros((N,))
  for jdx, (xi, yi, wxi, wyi) in enumerate(zip(X, Y, wx, wy)):
    wi = np.sum(wxi) * np.sum(wyi)
    k = wi * kernel(xi, yi)
    if np.isclose(np.abs(wi), 0.0):
      continue
    bo[jdx] = k
    for idx, (omegai, bi) in enumerate(zip(W, b)):
      M[jdx, idx] = wi * np.cos(np.dot(omegai, xi) + bi) * np.cos(np.dot(omegai, yi) + bi)

  a, _ = nnls(M, bo, maxiter=1000)

  # get new values
  new_idxs = (np.abs(a) > 3E-5)
  W = W[new_idxs]
  b = b[new_idxs]
  a = a[new_idxs]

  return a, W, b


def subsampled_dense_grid(d, D, gamma, deg=8):
  """sparse grid gaussian quadrature"""
  points, weights = np.polynomial.hermite.hermgauss(deg)
  points *= 2 * math.sqrt(gamma)
  weights /= math.sqrt(math.pi)
  # Now @weights must sum to 1.0, as the kernel value at 0 is 1.0
  subsampled_points = np.random.choice(points, size=(D, d), replace=True, p=weights)
  subsampled_weights = np.ones(D) / math.sqrt(D)
  return subsampled_points, subsampled_weights


def dense_grid(d, D, gamma, deg=8):
  """dense grid gaussian quadrature"""
  points, weights = np.polynomial.hermite.hermgauss(deg)
  points *= 2 * math.sqrt(gamma)
  weights /= math.sqrt(math.pi)
  points = np.atleast_2d(points).T
  return points, weights


def sparse_grid_map(n, d, sigma=1.0):
  """sparse GQ explicit map"""
  d, D = d, n
  gamma = 1/(2*sigma*sigma)
  points, weights = subsampled_dense_grid(d, D, gamma=gamma)
  def inner(x):
    prod = x @ points.T
    return np.hstack((weights * np.cos(prod), weights * np.sin(prod)))
  return inner


def dense_grid_map(n, d, sigma=1.0):
  """dense GQ explicit map"""
  d, D = d, n
  gamma = 1/(2*sigma*sigma)
  points, weights = dense_grid(d, D, gamma=gamma)
  def inner(x):
    prod = x @ points.T
    return np.hstack((weights * np.cos(prod), weights * np.sin(prod)))
  return inner


def quad_reweighted_map(n, X, Y, wx, wy, sigma=1.0):
  """build a RFF explicit feature map"""
  assert len(X) == len(Y)
  R = int(n / 2.0)
  D = X.shape[1]
  N = len(X)
  W, a = subsampled_dense_grid(D, R, gamma=1/(2*sigma*sigma))
  #W    = np.random.normal(loc=0, scale=(1.0/np.sqrt(sigma)), size=(R, D))
  b    = np.random.uniform(-np.pi, np.pi, size = R)
  #print(X.shape, W.shape)
  #b = np.arccos(-np.sqrt(np.cos(2*X.T.dot(W)) + 1)/np.sqrt(2.0))
  #print(b)

  # get ground truth kernel for training
  kernel = make_gaussian_kernel(sigma)

  # solve NNLS problem
  M = np.zeros((N, R))
  bo = np.zeros((N,))
  for jdx, (xi, yi, wxi, wyi) in enumerate(zip(X, Y, wx, wy)):
      k = kernel(xi, yi)
      bo[jdx] = k
      for idx, (omegai, ai, bi) in enumerate(zip(W, a, b)):
        M[jdx, idx] = np.cos(np.dot(omegai, xi - yi)) 

  a, _ = nnls(M, bo, maxiter=1000)

  # get new values
  new_idxs = (np.abs(a) > 1E-7)
  W = W[new_idxs]
  b = b[new_idxs]
  a = a[new_idxs]

  return a, W, b


class ReweightedRFFObservable(SymbolicObservable):
    def __init__(self, dimension, num_features, gamma, X, Y, wx, wy, feat_type='rff'):
        self.dimension, self.num_features = dimension, num_features
        n = num_features
        if feat_type == 'quad':
          self.a, self.W, self.b = quad_reweighted_map(n, X, Y, wx, wy, np.sqrt(1/(2*gamma)))
        elif feat_type == 'rff':
          self.a, self.W, self.b = rff_reweighted_map(n, X, Y, wx, wy, np.sqrt(1/(2*gamma)))
        else:
          raise ValueError('feat_type must be quad or rff')
        
        # make sympy variables for each of the state dimensions
        self.variables = [f'x{i}' for i in range(self.dimension)]
        self._observables = sp.symbols(self.variables)
        X = sp.Matrix(self._observables)
        
        # build observables sympy expressions from self.weights from self.weights, x.T @ points
        self.expressions = []
        for ai, wi, bi in zip(self.a, self.W, self.b):
            self.expressions.append(np.sqrt(ai) * sp.cos(X.dot(wi)))
            self.expressions.append(np.sqrt(ai) * sp.sin(X.dot(wi)))

        super(ReweightedRFFObservable, self).__init__(self.variables, self.expressions) 