import os
import sys
import numpy as np
from .Density_Ratio_discrete import Density_Ratio_discrete, Density_Ratio_discounted
from .Q_learning import Q_learning as Q_learning_class

def roll_out(state_num, env, policy, num_trajectory, truncate_size):
	SASR = []
	total_reward = 0.0
	frequency = np.zeros(state_num)
	for i_trajectory in range(num_trajectory):
		state = env.reset()
		sasr = []
		for i_t in range(truncate_size):
			#env.render()
			p_action = policy[state, :]
			action = np.random.choice(p_action.shape[0], 1, p = p_action)[0]
			next_state, reward = env.step(action)

			sasr.append( (state, action, next_state, reward) )
			frequency[state] += 1
			total_reward += reward
			#print env.state_decoding(state)
			#a = input()

			state = next_state
		SASR.append(sasr)
	return SASR, frequency, total_reward/(num_trajectory * truncate_size)

def train_density_ratio(SASR, policy0, policy1, den_discrete, gamma):
	for sasr in SASR:
		discounted_t = 1.0
		initial_state = sasr[0,0]
		for state, action, next_state, reward in sasr:
			discounted_t *= gamma
			policy_ratio = policy1[state, action]/policy0[state, action]
			den_discrete.feed_data(state, next_state, initial_state, policy_ratio, discounted_t)
		den_discrete.feed_data(-1, initial_state, initial_state, 1, 1-discounted_t)
		
	x, w = den_discrete.density_ratio_estimate()
	return x, w

def off_policy_evaluation_density_ratio(SASR, policy0, policy1, density_ratio, gamma):
	total_reward = 0.0
	self_normalizer = 0.0
	for sasr in SASR:
		discounted_t = 1.0
		for state, action, next_state, reward in sasr:
			policy_ratio = policy1[state, action]/policy0[state, action]
			total_reward += density_ratio[state] * policy_ratio * reward * discounted_t
			self_normalizer += density_ratio[state] * policy_ratio * discounted_t
			discounted_t *= gamma
	return total_reward / self_normalizer

def on_policy(SASR, gamma):
	total_reward = 0.0
	self_normalizer = 0.0
	for sasr in SASR:
		discounted_t = 1.0
		for state, action, next_state, reward in sasr:
			total_reward += reward * discounted_t
			self_normalizer += discounted_t
			discounted_t *= gamma
	return total_reward / self_normalizer

def importance_sampling_estimator(SASR, policy0, policy1, gamma):
	mean_est_reward = 0.0
	for sasr in SASR:
		log_trajectory_ratio = 0.0
		total_reward = 0.0
		discounted_t = 1.0
		self_normalizer = 0.0
		for state, action, next_state, reward in sasr:
			log_trajectory_ratio += np.log(policy1[state, action]) - np.log(policy0[state, action])
			total_reward += reward * discounted_t
			self_normalizer += discounted_t
			discounted_t *= gamma
		avr_reward = total_reward / self_normalizer
		mean_est_reward += avr_reward * np.exp(log_trajectory_ratio)
	mean_est_reward /= len(SASR)
	return mean_est_reward

def importance_sampling_estimator_stepwise(SASR, policy0, policy1, gamma):
	mean_est_reward = 0.0
	for sasr in SASR:
		step_log_pr = 0.0
		est_reward = 0.0
		discounted_t = 1.0
		self_normalizer = 0.0
		for state, action, next_state, reward in sasr:
			step_log_pr += np.log(policy1[state, action]) - np.log(policy0[state, action])
			est_reward += np.exp(step_log_pr)*reward*discounted_t
			self_normalizer += discounted_t
			discounted_t *= gamma
		est_reward /= self_normalizer
		mean_est_reward += est_reward
	mean_est_reward /= len(SASR)
	return mean_est_reward

def weighted_importance_sampling_estimator(SASR, policy0, policy1, gamma):
	total_rho = 0.0
	est_reward = 0.0
	for sasr in SASR:
		total_reward = 0.0
		log_trajectory_ratio = 0.0
		discounted_t = 1.0
		self_normalizer = 0.0
		for state, action, next_state, reward in sasr:
			log_trajectory_ratio += np.log(policy1[state, action]) - np.log(policy0[state, action])
			total_reward += reward * discounted_t
			self_normalizer += discounted_t
			discounted_t *= gamma
		avr_reward = total_reward / self_normalizer
		trajectory_ratio = np.exp(log_trajectory_ratio)
		total_rho += trajectory_ratio
		est_reward += trajectory_ratio * avr_reward

	avr_rho = total_rho / len(SASR)
	return est_reward / avr_rho/ len(SASR)

def weighted_importance_sampling_estimator_stepwise(SASR, policy0, policy1, gamma):
	Log_policy_ratio = []
	REW = []
	for sasr in SASR:
		log_policy_ratio = []
		rew = []
		discounted_t = 1.0
		self_normalizer = 0.0
		for state, action, next_state, reward in sasr:
			log_pr = np.log(policy1[state, action]) - np.log(policy0[state, action])
			if log_policy_ratio:
				log_policy_ratio.append(log_pr + log_policy_ratio[-1])
			else:
				log_policy_ratio.append(log_pr)
			rew.append(reward * discounted_t)
			self_normalizer += discounted_t
			discounted_t *= gamma
		Log_policy_ratio.append(log_policy_ratio)
		REW.append(rew)
	est_reward = 0.0
	rho = np.exp(Log_policy_ratio)
	#print 'rho shape = {}'.format(rho.shape)
	REW = np.array(REW)
	for i in range(REW.shape[0]):
		est_reward += np.sum(rho[i]/np.mean(rho, axis = 0) * REW[i])/self_normalizer
	return est_reward/REW.shape[0]


def Q_learning(env, num_trajectory, truncate_size, temperature = 2.0):
	agent = Q_learning_class(n_state, n_action, 0.01, 0.99)

	state = env.reset()
	for k in range(20):
		print('Training for episode {}'.format(k))
		for i in range(50):
			for j in range(5000):
				action = agent.choose_action(state, temperature)
				next_state, reward = env.step(action)
				agent.update(state, action, next_state, reward)
				state = next_state
		pi = agent.get_pi(temperature)
		np.save('taxi-policy/pi{}.npy'.format(k), pi)
		SAS, f, avr_reward = roll_out(n_state, env, pi, num_trajectory, truncate_size)
		print('Episode {} reward = {}'.format(k, avr_reward))
		heat_map(length, f, env, 'heatmap/pi{}.pdf'.format(k))

def heat_map(length, f, env, filename):
	p_matrix = np.zeros([length, length], dtype = np.float32)
	for state in range(env.n_state):
		x,y,_,_ = env.state_decoding(state)
		#x,y = env.state_decoding(state)
		p_matrix[x,y] = f[state]
	p_matrix = p_matrix / np.sum(p_matrix)
	
	sns.heatmap(p_matrix, cmap="YlGnBu")#, vmin = 0.0, vmax = 0.07)
	ppPDF = PdfPages(filename)
	ppPDF.savefig()
	ppPDF.close()
	plt.clf()

def model_based(n_state, n_action, SASR, pi, gamma):
	T = np.zeros([n_state, n_action, n_state], dtype = np.float32)
	R = np.zeros([n_state, n_action], dtype = np.float32)
	R_count = np.zeros([n_state, n_action], dtype = np.int32)
	for sasr in SASR:
		for state, action, next_state, reward in sasr:
			T[state, action, next_state] += 1
			R[state, action] += reward
			R_count[state, action] += 1
	d0 = np.zeros([n_state, 1], dtype = np.float32)

	for state in SASR[:,0,0].flat:
		d0[state, 0] += 1.0
	t = np.where(R_count > 0)
	t0 = np.where(R_count == 0)
	R[t] = R[t]/R_count[t]
	R[t0] = np.mean(R[t])
	T = T + 1e-9	# smoothing
	T = T/np.sum(T, axis = -1)[:,:,None]
	Tpi = np.zeros([n_state, n_state])
	for state in range(n_state):
		for next_state in range(n_state):
			for action in range(n_action):
				Tpi[state, next_state] += T[state, action, next_state] * pi[state, action]
	dt = d0/np.sum(d0)
	dpi = np.zeros([n_state, 1], dtype = np.float32)
	truncate_size = SASR.shape[1]
	discounted_t = 1.0
	self_normalizer = 0.0
	for i in range(truncate_size):
		dpi += dt * discounted_t
		if i < 50:
			dt = np.dot(Tpi.T,dt)
		self_normalizer += discounted_t
		discounted_t *= gamma
	dpi /= self_normalizer
	Rpi = np.sum(R * pi, axis = -1)
	return np.sum(dpi.reshape(-1) * Rpi)
	
