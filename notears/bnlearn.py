import numpy as np
import os
from notears import utils, linear, noleaks

def read_tsv(filename, priv_size, pub_size):
    with open(filename) as f:
        headers = [h[1:-1] for h in f.readline().strip().split('\t')]
        discrete2num = {h:{} for h in headers}
        discrete2idx = {h:0 for h in headers}
        frequencies = {h:{} for h in headers}
        lines = f.readlines()
        X = []
        
        for line in lines:
            point = line.strip().split('\t')
            num_point = []
            for idx, val in enumerate(point):
                col = headers[idx]
                if val not in discrete2num[col]:
                    mapping = np.ceil(discrete2idx[col] / 2)
                    if discrete2idx[col] % 2 == 0:
                        mapping *= -1
                    discrete2num[col][val] = discrete2idx[col]
                    discrete2idx[col] += 1
                num_val = discrete2num[col][val]
                if num_val not in frequencies[col]:
                    frequencies[col][num_val] = 1
                else:
                    frequencies[col][num_val] += 1
                num_point.append(num_val)
            X.append(num_point)
        X = np.array(X, dtype=np.float)
    
    X = discrete2continuous(X, headers, frequencies) * 20
    X = X - np.mean(X, axis=0, keepdims=True)
    priv_X = X[:priv_size]
    if pub_size != 0: pub_X = X[priv_size:priv_size+pub_size]
    else: pub_X = None
    return priv_X, pub_X, headers

def discrete2continuous(X, headers, frequencies):
    data_size = X.shape[0]
    models = {h:create_model(frequencies[h]) for h in headers}
    for row_idx in range(data_size):
        for col_idx, h in enumerate(headers):
            val = models[h](X[row_idx][col_idx])
            X[row_idx][col_idx] = val
    return X

def normal(x, mu, sig):
    return 1. / (np.sqrt(2 * np.pi) * sig) * np.exp(-0.5 * np.square(x - mu) / np.square(sig))


def trunc_normal(x, mu, sig, bounds=None):
    if bounds is None: 
        bounds = (-np.inf, np.inf)

    norm = normal(x, mu, sig)
    norm[x < bounds[0]] = 0
    norm[x > bounds[1]] = 0

    return norm


def sample_trunc(n, mu, sig, bounds=None):
    """ Sample `n` points from truncated normal distribution """
    x = np.linspace(mu - 5. * sig, mu + 5. * sig, 10000)
    y = trunc_normal(x, mu, sig, bounds)
    y_cum = np.cumsum(y) / y.sum()

    yrand = np.random.rand(n)
    sample = np.interp(yrand, y_cum, x)
    return sample

def create_model(freq):
    total = np.sum([freq[val] for val in freq])
    prob_dist = [(freq[i]/total, i) for i in range(len(freq))]
    prob_dist.sort(key=lambda x:-x[0]) # desceding order
    curr = 0
    left = {}
    right = {}
    for prob in prob_dist:
        left[prob[1]] = curr
        curr += prob[0]
        right[prob[1]] = curr
    def sample(val):
        mu = (right[val] - left[val]) / 2 + left[val]
        return mu
        sigma = (right[prob[1]] - left[prob[1]]) / 6
        return sample_trunc(1, mu, sigma, [left[val], right[val]])[0]
    return sample


def read_dag(filename, headers):
    adj_mat = np.zeros((len(headers), len(headers)))
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            if line.strip() == "": continue
            edge = line.strip().split("->")
            start = headers.index(edge[0])
            end = headers.index(edge[1])
            adj_mat[start][end] = 1
    return adj_mat

def get_dataset(dataset, data_size, has_public_data=True):
    tsv_file = os.path.join("bnlearn/dag_data/", f"{dataset}.tsv")
    dag_file = os.path.join("bnlearn/dag/", f"{dataset}.txt")
    if has_public_data:
        priv_X, pub_X, headers = read_tsv(tsv_file, data_size, int(data_size * 0.01))
    else: 
        priv_X, pub_X, headers = read_tsv(tsv_file, data_size, 0)
    adj_mat = read_dag(dag_file, headers)

    return priv_X, pub_X, adj_mat

if __name__ == '__main__':
    # subsample_size = 5000
    # datasets = ['asia', 'cancer', 'earthquake', 'survey', 'sachs', 'child', 'alarm', 'insurance', 'barley', 'mildew', 'water']
    # import time
    # for dataset in datasets:
    #     priv_X, pub_X, B_true = get_dataset(dataset, subsample_size, True)
    #     delta = 1e-4
    #     priv_config = noleaks.PrivConfiguration(0.1, delta)
    #     start = time.time()
    #     W_est = noleaks.noleaks(priv_X, priv_config, pub_X, is_priv=False) #linear.notears_linear(priv_X, lambda1=0.1, loss_type='l2', w_threshold=-1) #
    #     print(dataset, time.time() - start)
    # exit()
    # W_est = linear.notears_linear(priv_X, lambda1=0.1, loss_type='l2', w_threshold=-1)0

    subsample_size = 5000
    dataset = "asia"
    priv_X, pub_X, B_true = get_dataset(dataset, subsample_size, True)
    epsilon_list = np.logspace(-2.5, 0.5, num=40, base=10)
    epsilon_list = [1]
    f = open("out.txt", "w")
    results = [0.1]
    for epsilon in epsilon_list:
        delta = 1e-4
        priv_config = noleaks.PrivConfiguration(epsilon, delta)
        W_est = noleaks.noleaks(priv_X, priv_config, pub_X, is_priv=False) #linear.notears_linear(priv_X, lambda1=0.1, loss_type='l2', w_threshold=-1) #
        total = priv_config.privacy_amplification(subsample_size/100_000)
        # if total < 0.05: continue
        best = 0
        best_all = None
        for i in range(2000):
            W_curr = np.copy(W_est)
            W_curr[np.abs(W_curr) < i/1000] = 0
            acc = utils.count_skeleton_f1(B_true, W_curr != 0)
            if best < acc["ske-f1"] and acc["dag"]:
                best = acc["ske-f1"]
                best_all = acc
        results.append((total, best))
        if total > 10: break
    results.sort(key=lambda x: x[0])
    print(results)
    for total, best in results:
        print(total, ", " , best, file=f)
