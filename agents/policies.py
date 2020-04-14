import numpy as np
import math

class EpsilonGreedyPolicy:
    def __init__(self, epsilon, action_space):
        self.epsilon = epsilon
        self.action_space = action_space

    def get_action(self, estimator, state): 
        """
        Returns an action according to epsilon greedy policy
        """
        probs = self.prob(estimator, state)
        action = np.random.choice(range(self.action_space), p=probs)
        return action

    def prob(self, estimator, state):
        """Return the state action probability
        TODO: fix me...
        """
        max_value = -math.inf
        value_list = np.empty(self.action_space)   
        for action in range(self.action_space):
            value_list[action] = estimator.value(state, action)
            if value_list[action] > max_value:
                max_value = value_list[action]

        probs = np.full(self.action_space, self.epsilon / self.action_space)  
        max_actions = np.where(value_list == max_value)[0]
        for _action in max_actions:
            probs[_action] += (1 - self.epsilon) / len(max_actions)

        assert np.round(np.sum(probs), 3) == 1.0, "ERROR: The probabilities don't sum to 1!"

        return probs
        
class BoltzmannQPolicy:
    """Implement the Boltzmann Q Policy
    Boltzmann Q Policy builds a probability law on q values and returns
    an action selected randomly according to this law.
    The closer tau is to 0, greedier the policy is.
    The bigger tau is, the more uniform the policy is.
    """
    def __init__(self, tau=1., clip=(-500., 500.)):
        self.tau = tau
        self.clip = clip

    def get_action(self, estimator, state):
        """Return the selected action
        """
        probs = self.prob(estimator, state)
        action = np.random.choice(range(self.action_space), p=probs)
        return action

    def prob(self, estimator, state):
        """Return probability distribution
        """
        q_values = np.empty(self.action_space)   
        for _action in range(self.action_space):
            q_values[action] = estimator.value(state, _action)

        assert q_values.ndim == 1
        q_values = q_values.astype('float64')

        exp_values = np.exp(np.clip(q_values / self.tau, self.clip[0], self.clip[1]))
        probs = exp_values / np.sum(exp_values)

        assert np.round(np.sum(probs), 3) == 1.0, "ERROR: The probabilities don't sum to 1!"

        return probs

