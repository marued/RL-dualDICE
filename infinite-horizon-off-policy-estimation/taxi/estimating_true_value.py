import concurrent.futures
import multiprocessing
from functools import partial
import numpy as np
import matplotlib.pyplot as plt

from run_exp import *
from run_graphs import run_seeds_in_parallel

def on_policy_exe(n_state, n_action, env, pi_target, nt, ts, gm, k):
    np.random.seed(k)
    SASR, _, _ = roll_out(n_state, env, pi_target, nt, ts)
    res = on_policy(np.array(SASR), gm)

    return res, k

if __name__ == "__main__":
    estimator_names = ['On Policy']

    np.load
    
    # environment
    length = 5
    env = taxi(length)
    n_state = env.n_state
    n_action = env.n_action

    # Policies
    pi_target = np.load(os.getcwd() + '/infinite-horizon-off-policy-estimation/taxi/taxi-policy/pi19.npy')
    pi_behavior = np.load(os.getcwd() + '/infinite-horizon-off-policy-estimation/taxi/taxi-policy/pi18.npy')

    # Sampling vars
    ts = 400 # truncate_size
    nt = 100000
    gm = 0.995
    nb_seeds = 2

    lam_fct = partial(on_policy_exe, n_state, n_action, env, pi_target, nt, ts, gm)
    results = run_seeds_in_parallel(None, lam_fct, estimator_names, nb_seeds=nb_seeds)

    np.save( os.getcwd() + "/result/onpolicy_truevalue_estimate_{}_{}_{}.npy".format(nt, ts, gm), results )