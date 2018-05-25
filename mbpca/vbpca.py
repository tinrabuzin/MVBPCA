import numpy as np
from numpy.linalg import inv
from scipy.special import digamma, gammaln
from scipy.linalg import orth


class VBPCA:

    def __init__(self, t, m=1):
        self.N = t.shape[1]
        self.D = t.shape[0]
        self.Q = self.D-1
        self.M = m
        self.t = t
        self.l_q = np.array([])

        self.q_x = X(self.N, self.Q, self.M)
        self.q_mu = Mu(self.D, 1e-3, self.M)
        self.q_w = W(self.D, self.Q, self.M)
        self.q_alpha = Alpha(self.D, self.Q, 1e-3, 1e-3, self.M)
        self.q_tau = Tau(self.N, self.D, 1e-3, 1e-3, self.M)

    def fit(self, max_iter=10000, min_iter=100, eps=1e-3, verbose=False, init=True):
        self.l_q = np.array([self.lower_bound()])
        if init:
            self.init()
        for iteration in range(max_iter):
            if verbose:
                print('=================== \n Iteration: %i\n Lower bound: %0.3f' % (iteration, self.l_q[iteration]))
            self.q_x.update(self.t, self.q_mu, self.q_w, self.q_tau)
            self.q_mu.update(self.t, self.q_x, self.q_w, self.q_tau)
            self.q_w.update(self.t, self.q_x, self.q_mu, self.q_tau, self.q_alpha)
            self.q_alpha.update(self.q_w)
            self.q_tau.update(self.t, self.q_x, self.q_w, self.q_mu)

            self.l_q = np.append(self.l_q, self.lower_bound())
            if iteration > min_iter and np.abs(self.l_q[-1] - self.l_q[-2]) < eps:
                print('===> Final lower bound: %0.3f' % self.l_q[-1])
                break

    def init(self):
        self.q_mu.mean[:, 0] = np.mean(self.t, axis=1)

    def lower_bound(self):
        l_bound = self._e_w() + self._e_mu() + self._e_alpha() + self._e_tau() + self._e_x() + self._e_t()
        l_bound -= self._h_w() + self._h_mu() + self._h_alpha() + self._h_tau() + self._h_x()
        return l_bound

    def _e_t(self):
        if self.M == 1:
            s_nm = np.ones(self.N)
        else:
            s_nm = self.q_s.s_nm
        s = -0.5*self.N*self.D*(np.log(2*np.pi) - (digamma(self.q_tau.a)-np.log(self.q_tau.b)))
        s -= 0.5*self.q_tau.e_tau*np.sum(s_nm*self.q_tau.square)
        return s

    def _e_x(self):
        if self.M == 1:
            s_nm = np.ones(self.N)
        else:
            s_nm = self.q_s.s_nm
        xtx = np.zeros((self.M, self.N))
        for m in range(self.M):
            xtx[m] = (np.einsum('ij, ij -> j', self.q_x.mean[m], self.q_x.mean[m]) + np.trace(self.q_x.cov[m]))
        return - 0.5 * np.sum(s_nm*xtx)

    def _e_w(self):
        s = 0
        for m in range(self.M):
            for i in range(self.Q):
                    s += self.D*(digamma(self.q_alpha.a[m, i]) - np.log(self.q_alpha.b[m, i])) \
                         - self.q_alpha.e_alpha[m, i] * np.diag(self.q_w.wtw[m])[i]
        return 0.5 * s

    def _e_mu(self):
        s_mu = 0
        for m in range(self.M):
            mu = self.q_mu.mean[:, [m]]
            cov_mu = self.q_mu.cov[m]
            s_mu = mu.T @ mu + np.trace(cov_mu)
        return 0.5*self.D*self.M*np.log(self.q_mu.beta) - 0.5 * 1e-3 * s_mu

    def _e_tau(self):
        a = self.q_tau.a_tau
        b = self.q_tau.b_tau
        return a*np.log(b) - gammaln(a) + (a - 1) *(digamma(self.q_tau.a) - np.log(self.q_tau.b)) - b * self.q_tau.e_tau

    def _e_alpha(self):
        a = self.q_alpha.a_alpha
        b = self.q_alpha.b_alpha
        s = 0
        for m in range(self.M):
            for i in range(self.Q):
                s += (a - 1)*(digamma(self.q_alpha.a[m, i])-np.log(self.q_alpha.b[m, i])) - b*self.q_alpha.e_alpha[m, i]
        return self.Q*self.M*(a*np.log(b) - gammaln(a)) + s

    def _h_x(self):
        if self.M == 1:
            s_nm = np.ones((self.M, self.N))
        else:
            s_nm = self.q_s.s_nm
        s_cov = 0
        for m in range(self.M):
                s_cov += np.sum(s_nm[m])*np.linalg.slogdet(self.q_x.cov[m])[1]
        return -0.5*(self.N*self.Q + s_cov)

    def _h_w(self):
        m = 0
        s_cov = 0
        for m in range(self.M):
            s_cov += np.linalg.slogdet(self.q_w.cov[m])[1]
        return -0.5*self.D*(self.M*self.Q + s_cov)

    def _h_mu(self):
        s_cov = 0
        for m in range(self.M):
            s_cov += np.linalg.slogdet(self.q_mu.cov[m])[1]
        return -0.5*(self.M*self.D + self.D*s_cov)

    def _h_tau(self):
        a = self.q_tau.a
        b = self.q_tau.b
        return np.log(b) - gammaln(a) + (a - 1) * digamma(a) - a

    def _h_alpha(self):
        s = 0

        a = self.q_alpha.a[0, 0]
        for m in range(self.M):
            s += np.sum(np.log(self.q_alpha.b[m]))
        return self.Q*self.M*((a-1)*digamma(a) - a - gammaln(a)) + s


class X:

    def __init__(self, n, q, m=1):
        self.N = n
        self.Q = q
        self.M = m

        self.mean = [np.zeros((self.Q, self.N)) for i in range(self.M)]
        self.cov = [np.eye(self.Q) for i in range(self.M)]

    def update(self, t, q_mu, q_w, q_tau):
        for m in range(self.M):
            e_w = q_w.mean[m]
            e_tau = q_tau.e_tau

            self.cov[m] = inv(np.eye(self.Q) + e_tau * q_w.wtw[m])
            self.mean[m] = e_tau * self.cov[m] @ e_w.T @ (t - q_mu.mean[:, [m]])


class Mu:

    def __init__(self, d, beta, m=1):
        self.M = m
        self.D = d
        self.beta = beta

        self.mean = np.random.normal(loc=0.0, scale=1e-3, size=self.D*self.M).reshape(self.D, self.M)
        self.cov = [np.eye(self.D)/self.beta for m in range(self.M)]

    def update(self, t, q_x, q_w, q_tau, q_s=None):
        #self.mean[:, 0] = np.mean(t,1)

        e_tau = q_tau.e_tau
        n = t.shape[1]

        if q_s is None:
            s_nm = np.ones((self.M, n))
        else:
            s_nm = q_s.s_nm
        for m in range(self.M):
            self.cov[m] = np.eye(self.D) / (self.beta + e_tau * np.sum(s_nm[m]))
            self.mean[:, [m]] = e_tau * self.cov[m] @ np.sum(s_nm[m] * (t - q_w.mean[m] @ q_x.mean[m]), axis=1)[:, np.newaxis]


class W:

    def __init__(self, d, q, m=1):
        self.D = d
        self.Q = q
        self.M = m
        self.mean = [100*orth(np.random.randn(self.D, self.Q)) for i in range(self.M)]
        self.cov = [np.eye(self.Q) for i in range(self.M)]
        self.wtw = [np.eye(self.Q) for i in range(self.M)]
        self.calc_wtw()

    def update(self, t, q_x, q_mu, q_tau, q_alpha, q_s=None):

        if q_s is None:
            s_nm = np.ones((self.M, t.shape[1]))
        else:
            s_nm = q_s.s_nm

        e_tau = q_tau.e_tau
        for m in range(self.M):
            mu = q_mu.mean[:, [m]]
            diff = t - mu
            x_m = q_x.mean[m]

            s = np.sum(np.einsum('in,jn->nij', s_nm[m] * x_m, diff), axis=0)
            s1 = np.einsum('ij,kj->jik', x_m, x_m) + q_x.cov[m]
            s1 = np.einsum('i,ilm->lm', s_nm[m], s1)

            self.cov[m] = inv(np.diag(q_alpha.e_alpha[m]) + e_tau * s1)
            self.mean[m] = (e_tau * self.cov[m] @ s).T
        self.calc_wtw()

    def calc_wtw(self):
        for m in range(self.M):
            self.wtw[m] = self.mean[m].T @ self.mean[m] + self.D * self.cov[m]


class Alpha:

    def __init__(self, d, q, a_alpha, b_alpha, m=1):
        self.D = d
        self.Q = q
        self.M = m
        self.a_alpha = a_alpha
        self.b_alpha = b_alpha

        self.a = self.a_alpha * np.ones((self.M, self.Q))
        self.b = self.b_alpha * np.ones((self.M, self.Q))
        self.e_alpha = self.a / self.b

    def update(self, q_w):
        for m in range(self.M):
            self.a[m] = (self.a_alpha + self.D / 2) * np.ones(self.Q)
            self.b[m] = self.b_alpha * np.ones(self.Q) + 0.5 * np.diag(q_w.wtw[m])
        self.e_alpha = self.a / self.b


class Tau:

    def __init__(self, n, d, a_tau, b_tau, m=1):
        self.a_tau = a_tau
        self.b_tau = b_tau
        self.N = n
        self.D = d
        self.M = m
        self.square = np.zeros((self.M, self.N))

        self.a = self.a_tau
        self.b = self.b_tau
        self.e_tau = self.a / self.b

    def update(self, t, q_x, q_w, q_mu, q_s=None):
        self.a = self.a_tau + self.D * self.N / 2
        s_n = np.zeros((self.M, self.N))
        if q_s is None:
            s_nm = np.ones((self.M, t.shape[1]))
        else:
            s_nm = q_s.s_nm

        for m in range(self.M):
            mu = q_mu.mean[:, [m]]
            e_w = q_w.mean[m]
            wTw = q_w.wtw[m]
            x = q_x.mean[m]
            s_n[m] = np.einsum('in,in->n', t, t) + (np.einsum('ij,ij->j', mu, mu) + np.trace(q_mu.cov[m]))
            s_n[m] += 2 * np.einsum('l,ln->n', np.einsum('ij,ik->k', mu, e_w), x)
            s_n[m] -= 2 * np.einsum('ij,ji->i', np.einsum('ij,il->jl', t, e_w), x)
            s_n[m] -= 2 * np.einsum('ij,ik->j', t, mu)
            temp = np.einsum('ij,kj->jik', x, x) + q_x.cov[m]
            temp = np.einsum('ij,kjm->kim', wTw, temp)
            s_n[m] += np.einsum('kll->k', temp)

        self.square = s_n.copy()
        self.b = self.b_tau + 0.5 * np.sum(s_nm * s_n)
        self.e_tau = self.a / self.b
