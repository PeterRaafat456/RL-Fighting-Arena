import numpy as np
import random

class QLearningAgent:
    def __init__(self, state_dim=(101, 101), n_actions=3, alpha=0.1, gamma=0.95,
                 epsilon=0.2, epsilon_min=0.01, epsilon_decay=0.995):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = np.zeros((state_dim[0], state_dim[1], n_actions))

    def act(self, state, eval_mode=False):
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        s0, s1 = state
        return int(np.argmax(self.q_table[s0, s1, :]))

    def update(self, state, action, reward, next_state, done):
        s0, s1 = state
        ns0, ns1 = next_state
        best_next = np.max(self.q_table[ns0, ns1, :]) if not done else 0
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[s0, s1, action]
        self.q_table[s0, s1, action] += self.alpha * td_error
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_q_values(self, state):
        s0, s1 = state
        return self.q_table[s0, s1, :]


class DoubleQLearningAgent:
    def __init__(self, state_dim=(101, 101), n_actions=3, alpha=0.1, gamma=0.95,
                 epsilon=0.2, epsilon_min=0.01, epsilon_decay=0.995):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q1 = np.zeros((state_dim[0], state_dim[1], n_actions))
        self.q2 = np.zeros((state_dim[0], state_dim[1], n_actions))

    def act(self, state, eval_mode=False):
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        s0, s1 = state
        q_avg = (self.q1[s0, s1, :] + self.q2[s0, s1, :]) / 2
        return int(np.argmax(q_avg))

    def update(self, state, action, reward, next_state, done):
        s0, s1 = state
        ns0, ns1 = next_state
        if np.random.rand() < 0.5:
            # Update Q1
            if done:
                target = reward
            else:
                best_action = np.argmax(self.q1[ns0, ns1, :])
                target = reward + self.gamma * self.q2[ns0, ns1, best_action]
            td_error = target - self.q1[s0, s1, action]
            self.q1[s0, s1, action] += self.alpha * td_error
        else:
            # Update Q2
            if done:
                target = reward
            else:
                best_action = np.argmax(self.q2[ns0, ns1, :])
                target = reward + self.gamma * self.q1[ns0, ns1, best_action]
            td_error = target - self.q2[s0, s1, action]
            self.q2[s0, s1, action] += self.alpha * td_error
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_q_values(self, state):
        s0, s1 = state
        return (self.q1[s0, s1, :] + self.q2[s0, s1, :]) / 2