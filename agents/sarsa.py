class SarsaTrace:
    def __init__(self, q_estimator, discount_factor=1.0):
        self.q_estimator = q_estimator
        self.discount_factor = discount_factor

    def update(self, env, policy, state, action):
        # Calculate next target
        next_state, reward, done, _ = env.step(action)
        next_action = policy(self.q_estimator, next_state) # on policy target, for q-learning (off-policy) this is max(q_values)
        next_Q = self.q_estimator.value(next_state, next_action)
        if not done:
            target = reward + self.discount_factor * next_Q
        else: # TODO test difference between this if and no if...
            target = reward

        # Update QValue estimator
        self.q_estimator.update(state, action, target)

        return next_state, next_action, reward, done

    def start_new_episode(self):
        self.q_estimator.reset_trace()

    def reset(self):
        self.q_estimator.reset()

class Sarsa:
    def __init__(self, q_estimator, discount_factor=1.0):
        self.q_estimator = q_estimator
        self.discount_factor = discount_factor

    def update(self, env, policy, state, action):
        # Calculate next target
        next_state, reward, done, _ = env.step(action)
        next_action = policy(self.q_estimator, next_state) # on policy target, for q-learning (off-policy) this is max(q_values)
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
        self.q_estimator.reset()
