
class NStepImportanceSampling:
    """
    Page 149 Sutton and Barto RL Book.
    """
    def __init__(self, behavior_policy, target_policy):
        self.ro = 1
        self.arguments = None
        self.behavior_policy = behavior_policy
        self.target_policy = target_policy

    def calculate(self, estimator):
        """
        Expecting to have a list of states and actions, tau, NStep and final time T in 
        NStepImportanceSampling.arguments.
        Implementation just like p.149 of Sutton and Barto (chapter 7.3)
        """
        self.ro = 1
        state_list, action_list, tau, n, T = self.arguments
        for i in range(tau + 1, min(tau + n-1, T-1) + 1):
            action, state = (action_list[i], state_list[i])
            self.ro *= self.target_policy.prob(estimator, state)[action] / self.behavior_policy.prob(estimator, state)[action]

        return self.ro

    def value(self, state, action):
        return self.ro

    def post_update(self):
        pass

class DualDICE:
    #TODO
    def __init__(self):
        pass

    def update(self):
        pass

    def value(self, state, action):
        pass

    def post_update(self):
        pass