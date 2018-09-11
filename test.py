import numpy as np

def batchnorm_forward(x, gamma, beta, eps):

  N, W, H = x.shape

  #step1: calculate mean
  mu = 1/N * np.sum(x, axis = 0)

  #step2: subtract mean vector of every trainings example
  xmu = x - mu

  #step3: following the lower branch - calculation denominator
  sq = xmu ** 2

  #step4: calculate variance
  var = 1/N * np.sum(sq, axis = 0)

  #step5: add eps for numerical stability, then sqrt
  sqrtvar = np.sqrt(var + eps)

  #step6: invert sqrtwar
  ivar = 1/sqrtvar

  #step7: execute normalization
  xhat = xmu * ivar

  #step8: Nor the two transformation steps
  gammax = gamma * xhat

  #step9
  out = gammax + beta

  #store intermediate
  cache = (xmu, ivar, xhat, gamma)

  return out, cache

def batchnorm_backward(dout, cache):

  N, W, H = dout.shape

  #get the dimensions of the input/output
  x_mu, inv_var, x_hat, gamma = cache

  # intermediate partial derivatives
  dxhat = dout * gamma

  #print(inv_var / N)

  # final partial derivatives
  dx = (1 / N) * inv_var * (N*dxhat - np.sum(dxhat, axis=0) 
    - x_hat*np.sum(dxhat*x_hat, axis=0))

  dbeta = np.sum(dout, axis=0)
  dgamma = np.sum(x_hat*dout, axis=0)

  return dx, dgamma, dbeta

def main():
  x = np.array([[[ 1,  2,  3],
        [ 1,  2,  3],
        [ 1,  2,  3]],

       [[ 0.1,  0.2,  0.3],
        [ 0.1,  0.2,  0.3],
        [ 0.1,  0.2,  0.3]]], dtype=np.float32)
  gamma = np.array([[[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]]], dtype=np.float32)
  beta = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype=np.float32)
  dout = np.array([[[ 1,  2,  3],
        [ 1,  2,  3],
        [ 1,  2,  3]],

       [[ 10,  20,  30],
        [ 10,  20,  30],
        [ 10,  20,  30]]], dtype=np.float32) 
  out, cache = batchnorm_forward(x, gamma, beta, 0.001)
  dx, dgamma, dbeta = batchnorm_backward(dout, cache)
  #print(out)
  print(dgamma)
  print(dbeta)

if __name__ == "__main__":
    main()