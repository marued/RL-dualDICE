import os
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import multiprocessing
import argparse

from infinite_horizon_off_policy_estimation.taxi.environment import taxi
from infinite_horizon_off_policy_estimation.taxi.experiments import roll_out
from run_tools import run_wrapper, run_seeds_in_parallel

def varying_number_trajectories(estimator_names, nt_list = [200, 500, 1000, 2000]):
    """
    Run multiple experiments that vary the number of trajectories
    """
    # environment
    length = 5
    env = taxi(length)
    n_state = env.n_state
    n_action = env.n_action

    # Policies
    alpha = 0.0 # mixture ratio
    pi_target = np.load(os.getcwd() + '/infinite_horizon_off_policy_estimation/taxi/taxi-policy/pi19.npy')
    pi_behavior = np.load(os.getcwd() + '/infinite_horizon_off_policy_estimation/taxi/taxi-policy/pi18.npy')
    pi_behavior = alpha * pi_target + (1-alpha) * pi_behavior

    # Sampling vars
    ts = 400 # truncate_size
    gm = 0.99 # gamma
    nb_seeds = 12

    results = np.zeros( (len(nt_list), len(estimator_names), nb_seeds) )
    for idx, nt in enumerate(nt_list):
        lam_fct = partial(run_wrapper, n_state, n_action, env, roll_out, estimator_names, pi_behavior, 
                          pi_target, nt, ts, gm)
        ret = run_seeds_in_parallel(int(multiprocessing.cpu_count() / 2), lam_fct, estimator_names, nb_seeds)
        results[idx, :, :] = ret

    return results


def varying_gamma(estimator_names, gm_list):
    """
    Run multiple experiments that vary the gamma values
    """
    # environment
    length = 5
    env = taxi(length)
    n_state = env.n_state
    n_action = env.n_action

    # Policies
    alpha = 0.0 # mixture ratio
    pi_target = np.load(os.getcwd() + '/infinite_horizon_off_policy_estimation/taxi/taxi-policy/pi19.npy')
    pi_behavior = np.load(os.getcwd() + '/infinite_horizon_off_policy_estimation/taxi/taxi-policy/pi18.npy')
    pi_behavior = alpha * pi_target + (1-alpha) * pi_behavior

    # Sampling vars
    ts = 400 # truncate_size
    nt = 1000
    nb_seeds = 12

    results = np.zeros( (len(gm_list), len(estimator_names), nb_seeds) )
    for idx, gm in enumerate(gm_list):
        lam_fct = partial(run_wrapper, n_state, n_action, env, roll_out, estimator_names, pi_behavior, 
                          pi_target, nt, ts, gm)
        ret = run_seeds_in_parallel(int(multiprocessing.cpu_count() / 2), lam_fct, estimator_names, nb_seeds)
        results[idx, :, :] = ret

    return results

def varying_target_mixture(estimator_names, alpha_list):
    """
    Run multiple experiments that vary the alpha values used to mix policies
    """
    # environment
    length = 5
    env = taxi(length)
    n_state = env.n_state
    n_action = env.n_action

    # Policies
    pi_target = np.load(os.getcwd() + '/infinite_horizon_off_policy_estimation/taxi/taxi-policy/pi19.npy')
    pi_behavior = np.load(os.getcwd() + '/infinite_horizon_off_policy_estimation/taxi/taxi-policy/pi18.npy')

    # Sampling vars
    ts = 400 # truncate_size
    nt = 1000
    gm = 0.99
    nb_seeds = 12

    results = np.zeros( (len(alpha_list), len(estimator_names), nb_seeds) )
    for idx, alpha in enumerate(alpha_list):
        pi_behavior = alpha * pi_target + (1-alpha) * pi_behavior
        lam_fct = partial(run_wrapper, n_state, n_action, env, roll_out, estimator_names, pi_behavior, 
                          pi_target, nt, ts, gm)
        ret = run_seeds_in_parallel(int(multiprocessing.cpu_count() / 2), lam_fct, estimator_names, nb_seeds)
        results[idx, :, :] = ret

    return results

def varying_trajectories_and_alpha(estimator_names, nt_list, alpha_list):
    """
    Experiment when varying both number of trajectories and policy mixture
    """
    # environment
    length = 5
    env = taxi(length)
    n_state = env.n_state
    n_action = env.n_action

    # Policies
    pi_target = np.load(os.getcwd() + '/infinite_horizon_off_policy_estimation/taxi/taxi-policy/pi19.npy')
    pi_behavior = np.load(os.getcwd() + '/infinite_horizon_off_policy_estimation/taxi/taxi-policy/pi18.npy')

    # Sampling vars
    ts = 400 # truncate_size
    gm = 0.995
    nb_seeds = 12

    results = np.zeros( (len(alpha_list), len(nt_list), len(estimator_names), nb_seeds) )
    for i, alpha in enumerate(alpha_list):
        for j, nt in enumerate(nt_list):
            pi_behavior = alpha * pi_target + (1-alpha) * pi_behavior
            lam_fct = partial(run_wrapper, n_state, n_action, env, roll_out, estimator_names, pi_behavior, 
                            pi_target, nt, ts, gm)
            ret = run_seeds_in_parallel(int(multiprocessing.cpu_count() / 2), lam_fct, estimator_names, nb_seeds)
            results[i, j, :, :] = ret

    return results

def varying_trajectories_and_alpha_distant_behavior_policy(estimator_names, nt_list, alpha_list):
    """
    Same as varying_trajectories_and_alpha, but we choose a behavior policy that is 
    much different from the target one.
    """
    # environment
    length = 5
    env = taxi(length)
    n_state = env.n_state
    n_action = env.n_action

    # Policies
    pi_target = np.load(os.getcwd() + '/infinite_horizon_off_policy_estimation/taxi/taxi-policy/pi19.npy')
    pi_behavior = np.load(os.getcwd() + '/infinite_horizon_off_policy_estimation/taxi/taxi-policy/pi13.npy')

    # Sampling vars
    ts = 400 # truncate_size
    gm = 0.995
    nb_seeds = 12

    results = np.zeros( (len(alpha_list), len(nt_list), len(estimator_names), nb_seeds) )
    for i, alpha in enumerate(alpha_list):
        for j, nt in enumerate(nt_list):
            pi_behavior = alpha * pi_target + (1-alpha) * pi_behavior
            lam_fct = partial(run_wrapper, n_state, n_action, env, roll_out, estimator_names, pi_behavior, 
                            pi_target, nt, ts, gm)
            ret = run_seeds_in_parallel(int(multiprocessing.cpu_count() / 2), lam_fct, estimator_names, nb_seeds)
            results[i, j, :, :] = ret

    return results

def varying_trajectories_and_length(estimator_names, nt_list, ts_list):
    # environment
    length = 5
    env = taxi(length)
    n_state = env.n_state
    n_action = env.n_action

    # Policies
    pi_target = np.load(os.getcwd() + '/infinite_horizon_off_policy_estimation/taxi/taxi-policy/pi19.npy')
    pi_behavior = np.load(os.getcwd() + '/infinite_horizon_off_policy_estimation/taxi/taxi-policy/pi18.npy')

    # Sampling vars
    alpha = 0.0
    gm = 0.995
    nb_seeds = 12

    results = np.zeros( (len(nt_list), len(ts_list), len(estimator_names), nb_seeds) )
    for i, nt in enumerate(nt_list):
        for j, ts in enumerate(ts_list):
            pi_behavior = alpha * pi_target + (1-alpha) * pi_behavior
            lam_fct = partial(run_wrapper, n_state, n_action, env, roll_out, estimator_names, pi_behavior, 
                            pi_target, nt, ts, gm)
            ret = run_seeds_in_parallel(int(multiprocessing.cpu_count() / 2), lam_fct, estimator_names, nb_seeds)
            results[i, j, :, :] = ret

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only_display_graphs", default=False, type=bool, 
                        help="Don't run new experiments, only display current saved ones when set to True. Default: False")
    args = parser.parse_args()

    estimator_names = ['On Policy', 'Density Ratio', 'Naive Average', 'IST', 'ISS', 'WIST', 'WISS', 'Model Based', 'DualDICE']
    
    if not os.path.exists(os.getcwd() + "/result"):
            os.mkdir(os.getcwd() + "/result")

    nt_list = [200, 500, 1000, 2000]
    gm_list = [0.999, 0.9, 0.8, 0.7, 0.5]
    alpha_list = [0.0, 0.2, 0.5, 0.7]
    alpha_list2 = [0.0, 0.33, 0.66]
    nt_list2 = [500, 1000, 2000, 3000, 5000]
    nt_list3 = [50, 100, 200, 400]
    ts_list3 = [50, 100, 200, 400]
    if not args.only_display_graphs:
        # Number of trajectories
        results = varying_number_trajectories(estimator_names, nt_list)
        np.save(os.getcwd() + '/result/varying_nb_trajectories_{}.npy'.format("_".join([str(i) for i in nt_list])), results)
        
        # Changing Gamma
        results = varying_gamma(estimator_names, gm_list)
        np.save(os.getcwd() + '/result/varying_gamma_{}.npy'.format("_".join([str(i) for i in gm_list])), results)

        # Changing Gamma
        results = varying_target_mixture(estimator_names, alpha_list)
        np.save(os.getcwd() + '/result/varying_target_alpha_{}.npy'.format("_".join([str(i) for i in alpha_list])), results)

        # Reproducing graph in dual dice for taxi
        results = varying_trajectories_and_alpha(estimator_names, nt_list2, alpha_list2)
        np.save(os.getcwd() + '/result/varying_alpha_and_nt_({})_({}).npy'.format(
            "_".join([str(i) for i in alpha_list2]),
            "_".join([str(i) for i in nt_list2])),
            results)
    
        # Reproducing graph in dual dice for taxi
        results = varying_trajectories_and_length(estimator_names, nt_list3, ts_list3)
        np.save(os.getcwd() + '/result/varying_nt_and_ts_({})_({}).npy'.format(
            "_".join([str(i) for i in nt_list3]),
            "_".join([str(i) for i in ts_list3])),
            results)

        estimator_names2 = ['On Policy', 'Density Ratio', 'Naive Average', 'IST', 'ISS', 'DualDICE']
        results = varying_trajectories_and_alpha_distant_behavior_policy(estimator_names2, nt_list2, alpha_list2)
        np.save(os.getcwd() + '/result/varying_alpha_and_nt_diff_behavior({})_({}).npy'.format(
                "_".join([str(i) for i in alpha_list2]),
                "_".join([str(i) for i in nt_list2])),
                results)

    # plot
    trajectory_results = np.load( os.getcwd() + '/result/varying_nb_trajectories_{}.npy'.format("_".join([str(i) for i in nt_list])) )
    gamma_results = np.load( os.getcwd() + '/result/varying_gamma_{}.npy'.format("_".join([str(i) for i in gm_list])) )
    misture_policy_results = np.load( os.getcwd() + '/result/varying_target_alpha_{}.npy'.format("_".join([str(i) for i in alpha_list])) )
    alpha_trajectories_results = np.load(os.getcwd() + '/result/varying_alpha_and_nt_diff_behavior({})_({}).npy'.format(
        "_".join([str(i) for i in alpha_list2]),
        "_".join([str(i) for i in nt_list2])))
    nt_and_ts_results = np.load(os.getcwd() + '/result/varying_nt_and_ts_({})_({}).npy'.format(
        "_".join([str(i) for i in nt_list3]),
        "_".join([str(i) for i in ts_list3])))

    trajectory_results_min = trajectory_results.min(axis=-1)
    trajectory_results_mean = trajectory_results.mean(axis=-1)
    trajectory_results_max = trajectory_results.max(axis=-1)

    gamma_results_min = gamma_results.min(axis=-1)
    gamma_results_mean = gamma_results.mean(axis=-1)
    gamma_results_max = gamma_results.max(axis=-1)

    misture_policy_results_min = misture_policy_results.min(axis=-1)
    misture_policy_results_mean = misture_policy_results.mean(axis=-1)
    misture_policy_results_max = misture_policy_results.max(axis=-1)

    alpha_trajectories_min = alpha_trajectories_results.min(axis=-1)
    alpha_trajectories_mean = alpha_trajectories_results.mean(axis=-1)
    alpha_trajectories_max = alpha_trajectories_results.max(axis=-1)

    nt_and_ts_min = nt_and_ts_results.min(axis=-1)
    nt_and_ts_mean = nt_and_ts_results.mean(axis=-1)
    nt_and_ts_max = nt_and_ts_results.max(axis=-1)

    # Plot number of trajectories
    plt.plot(nt_list, trajectory_results_mean[:, 0],
            nt_list, trajectory_results_mean[:, -1])
    plt.fill_between(nt_list, trajectory_results_min[:, 0], trajectory_results_max[:, 0], color='pink', alpha=0.5)
    plt.fill_between(nt_list, trajectory_results_min[:, -1], trajectory_results_max[:, -1], color='grey', alpha=0.5)
    plt.show()

    # Plot gamma changes
    plt.plot(gm_list, gamma_results_mean[:, 0],
            gm_list, gamma_results_mean[:, -1])
    plt.fill_between(gm_list, gamma_results_min[:, 0], gamma_results_max[:, 0], color='grey', alpha=0.5)
    plt.fill_between(gm_list, gamma_results_min[:, -1], gamma_results_max[:, -1], color='pink', alpha=0.5)
    plt.show()

    # Policy change
    plt.plot(alpha_list, misture_policy_results_mean[:, 0],
            alpha_list, misture_policy_results_mean[:, -1])
    plt.fill_between(alpha_list, misture_policy_results_min[:, 0], misture_policy_results_max[:, 0], color='grey', alpha=0.5)
    plt.fill_between(alpha_list, misture_policy_results_min[:, -1], misture_policy_results_max[:, -1], color='pink', alpha=0.5)
    plt.show()

    # Reproducing graph
    plt.subplot(131)# alpha = 0.0
    plt.title('alpha = 0.0') 
    plt.xlabel('Number of trajectories') # Calculated from estimating_true_value.py
    plt.plot(nt_list2, alpha_trajectories_mean[0, :, 1] , 'green', label=estimator_names[1]) # Density Ratio
    plt.plot(nt_list2, alpha_trajectories_mean[0, :, 3], 'orange', label=estimator_names[3]) # IST
    plt.plot(nt_list2, alpha_trajectories_mean[0, :, 4] , 'blue', label=estimator_names[4]) # ISS
    plt.plot(nt_list2, alpha_trajectories_mean[0, :, -1], 'red', label=estimator_names[-1]) # dual diace
    plt.fill_between(nt_list2, alpha_trajectories_min[0, :, 3], alpha_trajectories_max[0, :, 3], color='orange', alpha=0.2)
    plt.fill_between(nt_list2, alpha_trajectories_min[0, :, 1], alpha_trajectories_max[0, :, 1], color='green', alpha=0.2)
    plt.fill_between(nt_list2, alpha_trajectories_min[0, :, 4], alpha_trajectories_max[0, :, 4], color='blue', alpha=0.2)
    plt.fill_between(nt_list2, alpha_trajectories_min[0, :, -1], alpha_trajectories_max[0, :, -1], color='pink', alpha=0.5)
    plt.axhline(y=-0.165, color='black', linestyle="--")

    plt.subplot(132) # alpha = 0.33
    plt.title('alpha = 0.33') 
    plt.xlabel('Number of trajectories')
    plt.plot(nt_list2, alpha_trajectories_mean[1, :, 1] , 'green', label=estimator_names[1]) # Density Ratio
    plt.plot(nt_list2, alpha_trajectories_mean[1, :, 3], 'orange', label=estimator_names[3]) # IST
    plt.plot(nt_list2, alpha_trajectories_mean[1, :, 4] , 'blue', label=estimator_names[4]) # ISS
    plt.plot(nt_list2, alpha_trajectories_mean[1, :, -1], 'red', label=estimator_names[-1]) # dual diace
    plt.fill_between(nt_list2, alpha_trajectories_min[1, :, 3], alpha_trajectories_max[1, :, 3], color='orange', alpha=0.2)
    plt.fill_between(nt_list2, alpha_trajectories_min[1, :, 1], alpha_trajectories_max[1, :, 1], color='green', alpha=0.2)
    plt.fill_between(nt_list2, alpha_trajectories_min[1, :, 4], alpha_trajectories_max[1, :, 4], color='blue', alpha=0.2)
    plt.fill_between(nt_list2, alpha_trajectories_min[1, :, -1], alpha_trajectories_max[1, :, -1], color='pink', alpha=0.5)
    plt.axhline(y=-0.165, color='black', linestyle="--") # Calculated from estimating_true_value.py

    plt.subplot(133) # alpha = 0.66
    plt.title('alpha = 0.66') 
    plt.xlabel('Number of trajectories')
    plt.plot(nt_list2, alpha_trajectories_mean[2, :, 1] , 'green', label=estimator_names[1]) # Density Ratio
    plt.plot(nt_list2, alpha_trajectories_mean[2, :, 3], 'orange', label=estimator_names[3]) # IST
    plt.plot(nt_list2, alpha_trajectories_mean[2, :, 4] , 'blue', label=estimator_names[4]) # ISS
    plt.plot(nt_list2, alpha_trajectories_mean[2, :, -1], 'red', label=estimator_names[-1]) # dual diace
    plt.fill_between(nt_list2, alpha_trajectories_min[2, :, 3], alpha_trajectories_max[2, :, 3], color='orange', alpha=0.2)
    plt.fill_between(nt_list2, alpha_trajectories_min[2, :, 1], alpha_trajectories_max[2, :, 1], color='green', alpha=0.2)
    plt.fill_between(nt_list2, alpha_trajectories_min[2, :, 4], alpha_trajectories_max[2, :, 4], color='blue', alpha=0.2)
    plt.fill_between(nt_list2, alpha_trajectories_min[2, :, -1], alpha_trajectories_max[2, :, -1], color='pink', alpha=0.5)
    plt.axhline(y=-0.165, color='black', linestyle="--")
    plt.legend()
    plt.show()

    # Plot Figure 1 of dual_dice
    plt.subplot(221)# nt = 50
    plt.title('# trajectories = 50') 
    plt.plot(ts_list3, nt_and_ts_mean[0, :, 0], 'orange', label=estimator_names[0]) # On policy
    plt.plot(ts_list3, nt_and_ts_mean[0, :, 1] , 'green', label=estimator_names[1]) # Density Ratio
    plt.plot(ts_list3, nt_and_ts_mean[0, :, 3] , 'blue', label=estimator_names[3]) # ISS
    plt.plot(ts_list3, nt_and_ts_mean[0, :, -1], 'red', label=estimator_names[-1]) # dual diace
    plt.fill_between(ts_list3, nt_and_ts_min[0, :, 0], nt_and_ts_max[0, :, 0], color='orange', alpha=0.5)
    plt.fill_between(ts_list3, nt_and_ts_min[0, :, 1], nt_and_ts_max[0, :, 1], color='green', alpha=0.2)
    plt.fill_between(ts_list3, nt_and_ts_min[0, :, 4], nt_and_ts_max[0, :, 4], color='blue', alpha=0.2)
    plt.fill_between(ts_list3, nt_and_ts_min[0, :, -1], nt_and_ts_max[0, :, -1], color='pink', alpha=0.5)

    plt.subplot(222)# nt = 100
    plt.title('# trajectories = 100') 
    plt.plot(ts_list3, nt_and_ts_mean[1, :, 0], 'orange', label=estimator_names[0]) # On policy
    plt.plot(ts_list3, nt_and_ts_mean[1, :, 1] , 'green', label=estimator_names[1]) # Density Ratio
    plt.plot(ts_list3, nt_and_ts_mean[1, :, 3] , 'blue', label=estimator_names[3]) # ISS
    plt.plot(ts_list3, nt_and_ts_mean[1, :, -1], 'red', label=estimator_names[-1]) # dual diace
    plt.fill_between(ts_list3, nt_and_ts_min[1, :, 0], nt_and_ts_max[1, :, 0], color='orange', alpha=0.5)
    plt.fill_between(ts_list3, nt_and_ts_min[1, :, 1], nt_and_ts_max[1, :, 1], color='green', alpha=0.2)
    plt.fill_between(ts_list3, nt_and_ts_min[1, :, 4], nt_and_ts_max[1, :, 4], color='blue', alpha=0.2)
    plt.fill_between(ts_list3, nt_and_ts_min[1, :, -1], nt_and_ts_max[1, :, -1], color='pink', alpha=0.5)

    plt.subplot(223)# nt = 200
    plt.title('# trajectories = 200') 
    plt.plot(ts_list3, nt_and_ts_mean[2, :, 0], 'orange', label=estimator_names[0]) # On policy
    plt.plot(ts_list3, nt_and_ts_mean[2, :, 1] , 'green', label=estimator_names[1]) # Density Ratio
    plt.plot(ts_list3, nt_and_ts_mean[2, :, 3] , 'blue', label=estimator_names[3]) # ISS
    plt.plot(ts_list3, nt_and_ts_mean[2, :, -1], 'red', label=estimator_names[-1]) # dual diace
    plt.fill_between(ts_list3, nt_and_ts_min[2, :, 0], nt_and_ts_max[2, :, 0], color='orange', alpha=0.5)
    plt.fill_between(ts_list3, nt_and_ts_min[2, :, 1], nt_and_ts_max[2, :, 1], color='green', alpha=0.2)
    plt.fill_between(ts_list3, nt_and_ts_min[2, :, 4], nt_and_ts_max[2, :, 4], color='blue', alpha=0.2)
    plt.fill_between(ts_list3, nt_and_ts_min[2, :, -1], nt_and_ts_max[2, :, -1], color='pink', alpha=0.5)

    plt.subplot(224)# nt = 400
    plt.title('# trajectories = 400') 
    plt.plot(ts_list3, nt_and_ts_mean[3, :, 0], 'orange', label=estimator_names[0]) # On policy
    plt.plot(ts_list3, nt_and_ts_mean[3, :, 1] , 'green', label=estimator_names[1]) # Density Ratio
    plt.plot(ts_list3, nt_and_ts_mean[3, :, 3] , 'blue', label=estimator_names[3]) # ISS
    plt.plot(ts_list3, nt_and_ts_mean[3, :, -1], 'red', label=estimator_names[-1]) # dual diace
    plt.fill_between(ts_list3, nt_and_ts_min[3, :, 0], nt_and_ts_max[3, :, 0], color='orange', alpha=0.5)
    plt.fill_between(ts_list3, nt_and_ts_min[3, :, 1], nt_and_ts_max[3, :, 1], color='green', alpha=0.2)
    plt.fill_between(ts_list3, nt_and_ts_min[3, :, 4], nt_and_ts_max[3, :, 4], color='blue', alpha=0.2)
    plt.fill_between(ts_list3, nt_and_ts_min[3, :, -1], nt_and_ts_max[3, :, -1], color='pink', alpha=0.5)
    plt.legend()
    plt.show()


