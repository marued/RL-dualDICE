from agents.tile_encoder import IHT
import numpy as np
import types


class QEstimator:

    def __init__(self, env_space, action_space, alpha):
        self.alpha = alpha # learning rate
        self.feature_space_size = int(env_space * action_space) 
        self.nb_states = env_space
        self.nb_actions = action_space

        self.reset()

    def update(self, state, action, target, importance_sampling=1.0):
        """
        Description:
        Single update step of q-values. 

        Params:
            state: The q value state we want to update. Used as index.
            action: The q value action we want to update. Used as index.
            target: The target provided by the agent algorithm. For example, 
                SARA's target = REWARD + discount * q_value[next_state, next_action]
                Q-Learning's target =  REWARD + discount * max_a(q_value[next_state, :])
            importance_sampling (optional): If planning to use off-policy (SARSA), 
                specify the importance_sampling that will be multiplied with the q-value
                update. It's possible to pass in a function that takes state, action as arguments.
                Default: 1.0 
        """
        ims_value = importance_sampling
        if isinstance(importance_sampling, types.FunctionType):
            ims_value = importance_sampling(state, action)

        self.q_value[state, action] += self.alpha * ims_value *( target - self.q_value[state, action])

    def value(self, state, action):
        """
        Return the estimated value of an action state pair. 
        """
        return self.q_value[state, action]

    def reset(self):
        self.q_value = np.zeros((self.nb_states, self.nb_actions))


class QEstimatorTraceTileEncoding:

    def __init__(self, action_space, min_max, alpha, lam, discount_factor=1.0, nb_tilings=8, tiling_dim=8):
        self.action_space = action_space
        # Number of possible different features, it's possible to cap it as well...
        # In our case, 8 * 8 * 8 * 3 = 
        self.feature_space_size = int(nb_tilings * tiling_dim * tiling_dim * action_space) 
        self.nb_tilings = nb_tilings

        self.iht = IHT(self.feature_space_size)
        self.weights = np.zeros(self.feature_space_size)
        self.trace = np.zeros(self.feature_space_size)

        self.alpha = alpha # learning rate of weigths
        self.lam = lam # trace fade factor
        self.discount_factor = discount_factor # also used for trace fading. Fallows sarsa discount

        # scaling for tiling, since bounderies are sliced at integer values. Ex: tiling_dim / (max - min)
        # This will respect tiling_dim for each tile
        self.min_max_scale = [ tiling_dim / (mm[1] - mm[0]) for mm in min_max]

    def update(self, state, action, target):
        # Update trace
        features = self.get_state_action_features(state, action)
        positive_mask = np.zeros(self.feature_space_size, bool)
        positive_mask[features] = True
        self.trace[positive_mask] = 1
        self.trace[~positive_mask]  *= self.lam * self.discount_factor
        
        # Update weights
        delta = target - self.value(state, action) # target - estimation
        self.weights += self.alpha * delta * self.trace

    def get_state_action_features(self, state, action):
        """
        Expecting actions to be discrete set. So the action input is a integer index.
        """
        return tiles(self.iht, self.nb_tilings, 
                    [self.min_max_scale[idx] * s for idx, s in enumerate(state)], 
                    [action])

    def value(self, state, action):
        """
        Return the estimated value of an action state pair. To do so, we encoded the 
        sates in tiles. We use tiles and action as feature input for function estimation.
        """
        # We also use the action as a feature. either that, or we could use different weight vectors per action
        features = self.get_state_action_features(state, action)
        active_weights = self.weights[features]
        return np.sum(active_weights)

    def reset_trace(self):
        self.trace = np.zeros(self.feature_space_size)

    def reset(self):
        self.weights = np.zeros(self.feature_space_size)
        self.trace = np.zeros(self.feature_space_size)
