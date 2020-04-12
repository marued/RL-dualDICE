from agents.estimators import QEstimator
from agents.sarsa import Sarsa
from agents.policies import epsilon_greedy_policy
import gym
import numpy as np


def on_policy_run(env, policy, agent, nb_episodes, max_nb_steps):
    episode_returns = np.empty([nb_episodes])
    env._max_episode_steps = max_nb_steps
    for episode in range(nb_episodes):
        done = False
        ret = 0
        state = env.reset()
        action = policy(agent.q_estimator, state)
        agent.start_new_episode()
        for step in range(max_nb_steps):
            next_state, next_action, reward, done = agent.update(env, policy, state, action)
            ret += reward
            if done:
                break
            state = next_state
            action = next_action
        # End of episode
        episode_returns[episode] = ret
    # End of all runs
    return episode_returns

def off_policy_run(env, behavior_policy, target_policy, agent, nb_episodes, max_nb_steps):
    episode_returns = np.empty([nb_episodes])
    env._max_episode_steps = max_nb_steps
    for episode in range(nb_episodes):
        done = False
        ret = 0
        state = env.reset()
        action = behavior_policy(agent.q_estimator, state)
        agent.start_new_episode()
        for step in range(max_nb_steps):
            next_state, next_action, reward, done = agent.update(env, target_policy, state, action)
            ret += reward
            if done:
                break
            state = next_state
            action = next_action
        # End of episode
        episode_returns[episode] = ret
    # End of all runs
    return episode_returns

def test_run(env, policy, q_estimator):
    done = False
    state = env.reset()
    action = policy(q_estimator, state)
    total_return = 0
    while not done:
        env.render()
        next_state, reward, done, _ = env.step(action)
        state = next_state
        action = policy(q_estimator, state)
        total_return += reward
    print(total_return)


if __name__ == "__main__":
    
    env = gym.make('Taxi-v3')  

    discount_factor=0.999
    alpha = alpha=0.1
    basic_estimator = QEstimator(env.nS, env.nA, alpha, discount_factor)
    basic_sarsa = Sarsa(basic_estimator, discount_factor)

    eps_policy = epsilon_greedy_policy(0.3, env.nA)
    # max_policy = epsilon_greedy_policy(0.0, env.nA)
    episode_returns = on_policy_run(env, eps_policy, basic_sarsa, nb_episodes=1000, max_nb_steps=1000)

    test_run(env, eps_policy, basic_sarsa.q_estimator)