from notears import utils, linear, noleaks, gd
import numpy as np

if __name__ == '__main__':
    from notears import utils
    utils.set_random_seed(1)
    # import time
    # for d in [5, 10, 15, 20, 50]:
    #     subsample_size = 10000
    #     graph_type, sem_type = 'ER', 'gauss'
    #     B_true = utils.simulate_dag(d, d, graph_type)
    #     indegree = np.sum(B_true,axis=1)
    #     W_true = utils.simulate_parameter(B_true)
    #     delta = 1e-4
    #     priv_X = utils.simulate_linear_sem(W_true, subsample_size, sem_type)
    #     priv_config = noleaks.PrivConfiguration(.1, delta)
    #     start = time.time()
    #     W_est = noleaks.noleaks(priv_X, priv_config, None, is_priv=True)
    #     print(d, time.time() - start)

    subsample_size = 10000
    pub_size = 10_0000 * 0.001
    case = 1
    d = 20
    graph_type, sem_type = 'ER', 'gauss'
    B_true = utils.simulate_dag(d, d, graph_type)
    W_true = utils.simulate_parameter(B_true)
    print(W_true)

    priv_X = utils.simulate_linear_sem(W_true, subsample_size, sem_type)
    if case != 1:
        pub_X = utils.simulate_linear_sem(W_true, pub_size, sem_type)
    else:
        pub_X = None

    # epsilon =  3.5e-3
    epsilon_list = [3.5e-3]
    for epsilon in epsilon_list:
        delta = 2e-4
        priv_config = noleaks.PrivConfiguration(epsilon, delta)
        W_est = noleaks.noleaks(priv_X, priv_config, pub_X, is_priv=True)
        total = priv_config.privacy_amplification(subsample_size/100_000)
        best = 0
        best_all = None
        for i in range(2000):
            W_curr = np.copy(W_est)
            W_curr[np.abs(W_curr) < i/100] = 0
            acc = utils.count_skeleton_f1(B_true, W_curr != 0)
            if best < acc["ske-f1"] and acc["dag"]:
                best = acc["ske-f1"]
                best_all = acc
        print(total, ", " , best_all)