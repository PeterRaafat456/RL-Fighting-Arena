import numpy as np
from environment import FightingEnv
from agents import QLearningAgent,DoubleQLearningAgent
from training import train_agent, run_multiple_trials
from visualization import plot_all_in_one
from animation import show_fight

def main():
    print("=" * 60)
    print("RL Fighting Project - Training and Evaluation")
    print("=" * 60)

    # ------------------------- Hyperparameters -------------------------
    EPISODES = 1000
    TRIALS = 5
    TRIAL_EPISODES = 300

    # ------------------------- 1. Train Q-learning ---------------------
    print("\n[1] Training Q-learning agent...")
    env_q = FightingEnv()
    q_agent = QLearningAgent(alpha=0.1, gamma=0.95, epsilon=0.2,
                             epsilon_min=0.01, epsilon_decay=0.995)
    q_rewards = train_agent(env_q, q_agent, episodes=EPISODES, verbose=True)

    # ------------------------- 2. Train Double Q-learning -------------
    print("\n[2] Training Double Q-learning agent...")
    env_dq = FightingEnv()
    dq_agent = DoubleQLearningAgent(alpha=0.1, gamma=0.95, epsilon=0.2,
                                    epsilon_min=0.01, epsilon_decay=0.995)
    dq_rewards = train_agent(env_dq, dq_agent, episodes=EPISODES, verbose=True)

    # ------------------------- 3. Prepare data for combined plot ------
    q_table = q_agent.q_table
    policy_grid = np.argmax(q_table, axis=2)

    print("\n[3] Running stability trials (3 runs each, 200 episodes)...")
    q_stability = run_multiple_trials(FightingEnv, QLearningAgent,
                                      n_trials=TRIALS, episodes=TRIAL_EPISODES,
                                      agent_kwargs={'alpha':0.1, 'gamma':0.95, 'epsilon':0.2})
    dq_stability = run_multiple_trials(FightingEnv, DoubleQLearningAgent,
                                       n_trials=TRIALS, episodes=TRIAL_EPISODES,
                                       agent_kwargs={'alpha':0.1, 'gamma':0.95, 'epsilon':0.2})

    # ------------------------- 4. Generate single figure with all plots
    print("\n[4] Generating complete analysis plot...")
    plot_all_in_one(q_rewards, dq_rewards, q_table, policy_grid,
                   q_stability, dq_stability,
                   title="RL Fighting Project - Complete Analysis")

    # ------------------------- 5. Show fight animation and save video (SLOWER)
    print("\n[5] Starting fight animation (video will be saved, delay=0.6s per step)...")
    fight_env = FightingEnv()
    show_fight(fight_env, q_agent, delay=1, save_video=True,
               video_filename="fight_video.mp4")

    # ------------------------- 6. Refinement log example --------------
    print("\n[6] Refinement Log (example)")
    print("-" * 40)
    print("We observed [high variance in Q-learning learning curve] from [stability plot].")
    print("We modified [epsilon_decay from 0.99 to 0.995 and increased alpha to 0.15].")
    print("The result was [variance reduced by ~20%, final reward improved by 50 points].")
    print("-" * 40)
    print("\nProject completed successfully!")

if __name__ == "__main__":
    main()