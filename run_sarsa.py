from agents.estimators import QEstimator
from agents.sarsa import Sarsa, SarsaNStep
from agents.policies import EpsilonGreedyPolicy, BoltzmannQPolicy
from agents.imp_samplers import NStepImportanceSampling
import gym
import numpy as np


def on_policy_run(env, policy, agent, nb_episodes, max_nb_steps):
    episode_returns = np.empty([nb_episodes])
    env._max_episode_steps = max_nb_steps
    for episode in range(nb_episodes):
        done = False
        ret = 0
        state = env.reset()
        action = policy.get_action(agent.q_estimator, state)
        agent.start_new_episode()
        while not done:
            next_state, next_action, reward, done = agent.update(env, policy, state, action)
            ret += reward
            state = next_state
            action = next_action
        # End of episode
        episode_returns[episode] = ret
    # End of all runs
    return episode_returns

def off_policy_run(env, behavior_policy, target_policy, agent, nb_episodes, max_nb_steps):
    """
    Target policy not being used for Sarsa since it's part of the imortance sampler only (the estimator update)...
    """
    episode_returns = np.empty([nb_episodes])
    env._max_episode_steps = max_nb_steps
    for episode in range(nb_episodes):
        done = False
        ret = 0
        state = env.reset()
        action = behavior_policy.get_action(agent.q_estimator, state)
        agent.start_new_episode()
        while not done:
            next_state, next_action, reward, done = agent.update(env, behavior_policy, state, action)
            ret += reward
            state = next_state
            action = next_action
        # End of episode
        episode_returns[episode] = ret
    # End of all runs
    return episode_returns

def test_run(env, policy, q_estimator):
    done = False
    state = env.reset()
    action = policy.get_action(q_estimator, state)
    total_return = 0
    steps = 0
    while not done:
        env.render()
        next_state, reward, done, _ = env.step(action)
        state = next_state
        action = policy.get_action(q_estimator, state)
        total_return += reward
        steps += 1
    print("Return:", total_return, ", steps:", steps)


if __name__ == "__main__":
    
    # TODO take the modified infinite horizon version
    env = gym.make('Taxi-v3')  

    discount_factor=0.999
    alpha = alpha=0.1
    
    #####################################################
    # Basic Sarsa
    #####################################################
    basic_estimator = QEstimator(env.nS, env.nA, alpha)
    basic_sarsa = Sarsa(basic_estimator, discount_factor)
    eps_policy = EpsilonGreedyPolicy(0.3, env.nA)

    #episode_returns = on_policy_run(env, eps_policy, basic_sarsa, nb_episodes=1000, max_nb_steps=1001)
    #test_run(env, eps_policy, basic_sarsa.q_estimator)

    #####################################################
    # On policy n-step Sarsa
    #####################################################
    eps_policy = EpsilonGreedyPolicy(0.3, env.nA)
    n_step_estimator = QEstimator(env.nS, env.nA, alpha)
    n_step_sarsa = SarsaNStep(n_step_estimator, imp_sampler=None, n_step=2, discount_factor=discount_factor)

    #episode_returns = on_policy_run(env, eps_policy, n_step_sarsa, nb_episodes=1000, max_nb_steps=1001)
    #test_run(env, eps_policy, n_step_sarsa.q_estimator)

    #####################################################
    # Off policy n-step Sarsa with importance sampling
    #####################################################
    eps_policy = EpsilonGreedyPolicy(0.3, env.nA)
    max_policy = EpsilonGreedyPolicy(0.0, env.nA)
    n_step_sampler = NStepImportanceSampling(behavior_policy=eps_policy, target_policy=max_policy)
    n_step_estimator = QEstimator(env.nS, env.nA, alpha)
    n_step_sarsa = SarsaNStep(n_step_estimator, imp_sampler=n_step_sampler, n_step=2, discount_factor=0.95)

    episode_returns = off_policy_run(env, eps_policy, max_policy, n_step_sarsa, nb_episodes=10000, max_nb_steps=500)
    test_run(env, max_policy, n_step_sarsa.q_estimator)