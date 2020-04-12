import numpy as np

def epsilon_greedy_policy(epsilon, action_space):
    def policy(estimator, state): 
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

    return policy