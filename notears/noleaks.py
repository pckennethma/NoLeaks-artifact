import numpy as np
from numpy.core.numeric import zeros_like
import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.optimize import zeros
from notears.aGM import calibrateAnalyticGaussianMechanism

np.random.seed(0)


class PrivConfiguration:
    def __init__(self,  epsilon, delta):
        self.basic_epsilon = epsilon
        self.basic_delta = delta
        self.total_epsilon = .0
        self.total_delta = .0

        self.init_clipping_threshold = 1
        self.clipping_count = 0
        self.g_oracle_count = 0

        self.agm_sigma = {}
    
    def add_priv_budget(self, epsilon, delta):
        self.total_epsilon += epsilon
        self.total_delta += delta
    
    def add_basic_budget(self):
        self.add_priv_budget(self.basic_epsilon, self.basic_delta)
    
    def get_agm_sigma(self, sensitivity):
        if sensitivity not in self.agm_sigma:
            sigma = calibrateAnalyticGaussianMechanism(self.basic_epsilon, self.basic_delta, sensitivity)
            self.agm_sigma[sensitivity] = sigma
        return self.agm_sigma[sensitivity]
    
    def report_budget(self):
        print(F"epsilon: {self.total_epsilon}; delta: {self.total_delta}")
    
    def privacy_amplification(self, rate):
        print(F"amplified epsilon: {self.total_epsilon * rate}; amplified delta: {self.total_delta * rate}")
        return self.total_epsilon * rate


def noleaks(X, priv_config, pub_X=None, lambda1=0.1, max_iter=100, h_tol=1e-8, rho_max=1e+16, is_priv=True):
    """Solve min_W L(W; X) + lambda1 ‖W‖_1 s.t. h(W) = 0 using augmented Lagrangian.

    Args:
        X (np.ndarray): [n, d] sample matrix
        lambda1 (float): l1 penalty parameter
        loss_type (str): l2, logistic, poisson
        max_iter (int): max num of dual ascent steps
        h_tol (float): exit if |h(w_est)| <= htol
        rho_max (float): exit if rho >= rho_max
        w_threshold (float): drop edge if |weight| < threshold

    Returns:
        W_est (np.ndarray): [d, d] estimated DAG
    """

    def _loss(W):
        n, d = X.shape
        M = X @ W
        R = X - M
        loss = 0.5 / n * (R ** 2).sum()
        return loss
    
    def _dp_loss(W):
        n, d = pub_X.shape
        M = pub_X @ W
        R = pub_X - M
        loss = 0.5 / n * (R ** 2).sum()
        
        return loss

    def _G_loss(W):
        """Evaluate value and gradient of loss."""

        n, d = X.shape

        M = X @ W
        R = X - M
        
        G_loss = - 1.0 / n * X.T @ R

        return G_loss

    def _dp_G_loss(W):
        """Evaluate value and gradient of loss."""
        n, d = X.shape

        M = X @ W
        R = X - M
        
        G_loss = - 1.0 / n * X.T @ R

        clipping_threshold = 2
        l2_sensitivity = np.sqrt(d * (d - 1)) * clipping_threshold * 2 / n
        
        sigma = priv_config.get_agm_sigma(l2_sensitivity)

        G_loss_priv = np.clip(G_loss, -clipping_threshold, clipping_threshold) + np.random.normal(0, sigma, G_loss.shape)
        # if np.abs(np.mean(G_loss_priv)) < sigma: return np.zeros_like(G_loss_priv)
        np.fill_diagonal(G_loss_priv, 0)

        priv_config.clipping_count += 1
        priv_config.add_basic_budget()

        return G_loss_priv

    def _h(W):
        """Evaluate value and gradient of acyclicity constraint."""
        E = slin.expm(W * W)  # (Zheng et al. 2018)
        h = np.trace(E) - d
        # A different formulation, slightly faster at the cost of numerical stability
        # M = np.eye(d) + W * W / d  # (Yu et al. 2019)
        # E = np.linalg.matrix_power(M, d - 1)
        # h = (E.T * M).sum() - d
        return h

    def _G_h(W):
        """Evaluate value and gradient of acyclicity constraint."""
        E = slin.expm(W * W)  # (Zheng et al. 2018)
        # A different formulation, slightly faster at the cost of numerical stability
        # M = np.eye(d) + W * W / d  # (Yu et al. 2019)
        # E = np.linalg.matrix_power(M, d - 1)
        G_h = E.T * W * 2
        return G_h

    def _adj(w):
        """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
        return (w[:d * d] - w[d * d:]).reshape([d, d])

    def _dp_G_obj_func(w):
        """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
        W = _adj(w)
        G_loss = _dp_G_loss(W)
        h = _h(W)
        G_h = _G_h(W)
        G_smooth = G_loss + (rho * h + alpha) * G_h
        g_obj = np.concatenate((G_smooth + lambda1, - G_smooth + lambda1), axis=None)
        return g_obj
    
    def _G_obj_func(w):
        """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
        W = _adj(w)
        G_loss = _G_loss(W)
        h = _h(W)
        G_h = _G_h(W)
        G_smooth = G_loss + (rho * h + alpha) * G_h
        g_obj = np.concatenate((G_smooth + lambda1, - G_smooth + lambda1), axis=None)
        return g_obj
    
    def _obj_func(w):
        W = _adj(w)
        loss = _loss(W)
        h = _h(W)
        obj = loss 
        obj += 0.5 * rho * h * h 
        obj += alpha * h 
        obj += lambda1 * w.sum()
        return obj
    
    def _dp_obj_func(w):
        W = _adj(w)
        loss = _dp_loss(W)
        h = _h(W)
        obj = loss
        obj += 0.5 * rho * h * h 
        obj += alpha * h 
        obj += lambda1 * w.sum()
        return obj


    n, d = X.shape
    if not isinstance(pub_X, (np.ndarray, np.generic) ):
        subsample_size = 1000
        pub_X = X[:subsample_size]
        epsilon = np.log((n+1)/(n+1-subsample_size))
        delta = subsample_size/n
        priv_config.add_priv_budget(epsilon, delta)
    
    w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf  # double w_est into (w_pos, w_neg)

    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
    # X = X - np.mean(X, axis=0, keepdims=True)
    # pub_X = pub_X  - np.mean(X, axis=0, keepdims=True)
    for _ in range(max_iter):
        w_new, h_new = None, None
        while rho < rho_max:
            if is_priv:
                sol = sopt.minimize(_dp_obj_func, w_est, method='L-BFGS-B', jac=_dp_G_obj_func, bounds=bnds)
            else:
                sol = sopt.minimize(_obj_func, w_est, method='L-BFGS-B', jac=_G_obj_func, bounds=bnds)
            w_new = sol.x
            h_new = _h(_adj(w_new))
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            print("early stop", h, rho)
            break
    W_est = _adj(w_est)
    # np.savetxt('W_est.csv', W_est, delimiter=',')
    # W_est[np.abs(W_est) < w_threshold] = 0
    return W_est


if __name__ == '__main__':
    from notears import utils
    
    utils.set_random_seed(1)

    n, d, s0, graph_type, sem_type = 10_000, 50, 50, 'ER', 'gauss'
    B_true = utils.simulate_dag(d, s0, graph_type)
    W_true = utils.simulate_parameter(B_true)
    np.savetxt('W_true.csv', W_true, delimiter=',')

    X = utils.simulate_linear_sem(W_true, n, sem_type, normalize=False)
    X = X - np.mean(X, axis=0, keepdims=True)
    
    np.savetxt('X.csv', X, delimiter=',')

    epsilon = 5
    delta = 1e-3
    priv_config = PrivConfiguration(epsilon, delta, X)

    W_est = noleaks(X, priv_config)

    priv_config.report_budget()
    assert utils.is_dag(W_est)
    #np.savetxt('W_est.csv', W_est, delimiter=',')
    acc = utils.count_accuracy(B_true, W_est != 0)
    f1 = utils.count_f1(B_true, W_est != 0)
    ske_f1 = utils.count_skeleton_f1(B_true, W_est != 0)

    assert utils.is_dag(W_est)

    print(acc, f1, ske_f1)
    print(priv_config.g_oracle_count)
