import numpy as np

class epsilon_greedy_policy:
    def __init__(epsilon, action_space):
        self.epsilon = epsilon
        self.action_space = action_space

    def get_action(estimator, state): 
        """
        Returns an action according to epsilon greedy policy
        """
        if np.random.random() <= epsilon:
            return np.random.choice(action_space)
        else:
            value_list = np.empty(action_space)   
            for action in range(action_space):
                value_list[action] = estimator.value(state, action)
            
            return np.argmax(value_list)

    def prob(estimator, action, state):
        """
        return the state action probability
        """
        #TODO
