import numpy as np

def train_agent(env, agent, episodes=500, verbose=True):
    """
    Train a single agent on the given environment.
    Returns list of total rewards per episode.
    """
    rewards = []
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        rewards.append(total_reward)
        if verbose and (ep + 1) % 100 == 0:
            avg = np.mean(rewards[-100:])
            print(f"Episode {ep+1}/{episodes}, Avg Reward (last 100): {avg:.2f}")
    return rewards

def run_multiple_trials(env_class, agent_class, n_trials=3, episodes=200, agent_kwargs=None):
    """
    Run several independent trials for stability analysis.
    Returns a 2D numpy array of shape (n_trials, episodes) padded if needed.
    """
    if agent_kwargs is None:
        agent_kwargs = {}
    all_rewards = []
    for t in range(n_trials):
        print(f"Trial {t+1}/{n_trials}")
        env = env_class()
        agent = agent_class(**agent_kwargs)
        rewards = train_agent(env, agent, episodes=episodes, verbose=False)
        all_rewards.append(rewards)
    # Pad shorter lists to same length (in case of early termination)
    max_len = max(len(r) for r in all_rewards)
    padded = [r + [r[-1]] * (max_len - len(r)) for r in all_rewards]
    return np.array(padded)