import numpy as np
import concurrent.futures
import multiprocessing
from functools import partial
import argparse
import sys
import os

import infinite_horizon_off_policy_estimation.taxi.experiments as xp
from infinite_horizon_off_policy_estimation.taxi.environment import taxi
from dual_dice.policy import TabularPolicy
from dual_dice.transition_data import TrajectoryData
from dual_dice.algos.dual_dice import TabularDualDice

def dual_dice(n_state, n_action, SASR, pi1, gamma):
	"""No need for behavior policy
	"""
	# base = TabularPolicy(pi0) # no need for base policy
	target = TabularPolicy(pi1)
	SARS = np.array(SASR)[:,:,[0,1,3,2]]
	data = TrajectoryData(SARS)
	dual_dice_obj = TabularDualDice(n_state, n_action, gamma, solve_for_state_action_ratio=True)
	return dual_dice_obj.solve(data, target)

def run_experiment(n_state, n_action, SASR, pi0, pi1, gamma):
	
	den_discrete = xp.Density_Ratio_discounted(n_state, gamma)
	x, w = xp.train_density_ratio(SASR, pi0, pi1, den_discrete, gamma)
	x = x.reshape(-1)
	w = w.reshape(-1)

	est_DENR = xp.off_policy_evaluation_density_ratio(SASR, pi0, pi1, w, gamma)
	est_naive_average = xp.on_policy(SASR, gamma)
	est_IST = xp.importance_sampling_estimator(SASR, pi0, pi1, gamma)
	est_ISS = xp.importance_sampling_estimator_stepwise(SASR, pi0, pi1, gamma)
	est_WIST = xp.weighted_importance_sampling_estimator(SASR, pi0, pi1, gamma)
	est_WISS = xp.weighted_importance_sampling_estimator_stepwise(SASR, pi0, pi1, gamma)
	dual_dice_return = dual_dice(n_state, n_action, SASR, pi1, gamma)
	
	est_model_based = xp.model_based(n_state, n_action, SASR, pi1, gamma)

	return est_DENR, est_naive_average, est_IST, est_ISS, est_WIST, est_WISS, est_model_based, dual_dice_return

def run_wrapper(n_state, n_action, env, roll_out, estimator_name, pi_behavior, 
				pi_target, nt, ts, gm, k):
	res =  np.zeros((len(estimator_name)))
	np.random.seed(k)
	SASR0, _, _ = xp.roll_out(n_state, env, pi_behavior, nt, ts)
	res[1:] = run_experiment(n_state, n_action, np.array(SASR0), pi_behavior, pi_target, gm)
	
	np.random.seed(k)
	SASR, _, _ = xp.roll_out(n_state, env, pi_target, nt, ts)
	res[0] = xp.on_policy(np.array(SASR), gm)
	
	return res, k

def run_seeds_in_parallel(nb_processes, partial_fct, estimator_names, nb_seeds=12):
    seeds = range(nb_seeds)

    res = np.zeros((len(estimator_names), nb_seeds), dtype = np.float32)
    with concurrent.futures.ProcessPoolExecutor(nb_processes) as executor:
        for ret, k in executor.map(partial_fct, seeds):
            res[:, k] = ret

    for k in seeds:
        print('------seed = {}------'.format(k))
        for i in range(len(estimator_names)):
            print('  ESTIMATOR: '+estimator_names[i]+ ', rewards = {}'.format(res[i,k]))
        print('----------------------')
        sys.stdout.flush()
    
    #executor.terminate()
    return res

if __name__ == "__main__":
	"""
	Test if experiments work in paralel
	"""
	estimator_names = ['On Policy', 'Density Ratio', 'Naive Average', 'IST', 'ISS', 'WIST', 'WISS', 'Model Based', 'DualDICE']
	length = 5
	env = taxi(length)
	n_state = env.n_state
	n_action = env.n_action
	
	num_trajectory = 50
	truncate_size = 50
	gamma = 0.995

	parser = argparse.ArgumentParser(description='taxi environment')
	parser.add_argument('--nt', type = int, required = False, default = num_trajectory)
	parser.add_argument('--ts', type = int, required = False, default = truncate_size)
	parser.add_argument('--gm', type = float, required = False, default = gamma)
	args = parser.parse_args()

	behavior_ID = 4
	target_ID = 5
	
	pi_target = np.load(os.getcwd() + '/infinite_horizon_off_policy_estimation/taxi/taxi-policy/pi19.npy')
	alpha = 0.0 # mixture ratio
	nt = args.nt # num_trajectory
	ts = args.ts # truncate_size
	gm = args.gm # gamma
	pi_behavior = np.load(os.getcwd() + '/infinite_horizon_off_policy_estimation/taxi/taxi-policy/pi18.npy')

	pi_behavior = alpha * pi_target + (1-alpha) * pi_behavior

	res = np.zeros((len(estimator_names), 20), dtype = np.float32)

	# run experiments in paralel:
	seeds = range(3)
	lam_fct = partial(run_wrapper, n_state, n_action, env, xp.roll_out, estimator_names, pi_behavior, 
				pi_target, nt, ts, gm)
	with concurrent.futures.ProcessPoolExecutor(int(multiprocessing.cpu_count() / 4)) as executor:
		for ret, k in executor.map(lam_fct, seeds):
			res[:, k] = ret

	for k in seeds:
		print('------seed = {}------'.format(k))
		for i in range(len(estimator_names)):
			print('  ESTIMATOR: '+estimator_names[i]+ ', rewards = {}'.format(res[i,k]))
		print('----------------------')
		sys.stdout.flush()

	# Save results
	if not os.path.exists(os.getcwd() + "/result"):
		os.mkdir(os.getcwd() + "/result")
	np.save(os.getcwd() + '/result/nt={}ts={}gm={}.npy'.format(nt,ts,gm), res)
    