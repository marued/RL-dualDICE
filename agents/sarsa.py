import math
import numpy as np

class Sarsa:
    """
    Sarsa clasic or with traces version. To support tracing, simply pass a 
    Q-estimator that supports traces. The overall algorithm is the same for both.
    i.e. The targets are the same in both cases, it's just the way we update Q that 
    changes. 
    """
    def __init__(self, q_estimator, discount_factor=1.0):
        self.q_estimator = q_estimator
        self.discount_factor = discount_factor

    def update(self, env, policy, state, action):
        # Calculate next target
        next_state, reward, done, _ = env.step(action)
        next_action = policy.get_action(self.q_estimator, next_state) # on policy target, for q-learning (off-policy) this is max(q_values)
        next_Q = self.q_estimator.value(next_state, next_action)
        if not done:
            target = reward + self.discount_factor * next_Q
        else: # TODO test difference between this if and no if...
            target = reward

        # Update QValue estimator
        self.q_estimator.update(state, action, target)

        return next_state, next_action, reward, done

    def start_new_episode(self):
        """
        If we were to do a n_step sarsa, we would need to reset counter here
        """
        pass

    def reset(self):
        self.q_estimator.start_new_episode()


class SarsaNStep:
    """
    Implementation of Sutton and Barto p.149: Off-policy n-step Sarsa
    """
    def __init__(self, q_estimator, imp_sampler=None, n_step=2, discount_factor=1.0):
        self.q_estimator = q_estimator
        self.discount_factor = discount_factor
        self.imp_sampler = imp_sampler
        self.n_step = n_step
        self.reset()

    def start_new_episode(self):
        self.step_counter = 0
        self.T = math.inf
        self.actions = []
        self.states = []
        self.rewards = [0]

    def reset(self):
        self.q_estimator.reset()
        self.start_new_episode()

    def update(self, env, behavior_policy, state, action):
        # First action and state we want to keep. We return None to make sure 
        # not to add it twice.
        reward = None
        if state is not None and action is not None:
            self.states.append(state)
            self.actions.append(action)

        # Start of classic algorithm
        if self.step_counter < self.T:
            # Take a step
            next_state, reward, done, _ = env.step(self.actions[self.step_counter])
            self.rewards.append(reward)
            self.states.append(next_state)

            if done:
                self.T = self.step_counter + 1
            else:
                next_action = behavior_policy.get_action(self.q_estimator, next_state)
                self.actions.append(next_action)

        # tau is the time whose estimate is being updated
        tau = self.step_counter - self.n_step + 1 

        if tau >= 0:
            # Calculate target
            target = 0
            for i in range(tau + 1, min(tau + self.n_step, self.T) + 1):
                target += np.power(self.discount_factor, i - tau - 1) * self.rewards[i]
            if tau + self.n_step < self.T:
                next_q = self.q_estimator.value(self.states[tau + self.n_step], self.actions[tau + self.n_step])
                target += np.power(self.discount_factor, self.n_step) * next_q

            # Update
            if self.imp_sampler is not None:
                self.imp_sampler.arguments = (self.states, self.actions, tau, self.n_step, self.T) 
            self.q_estimator.update(self.states[tau], self.actions[tau], target, importance_sampling=self.imp_sampler)

        self.step_counter += 1
        return None, None, reward, done
