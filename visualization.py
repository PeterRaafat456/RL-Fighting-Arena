import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_all_in_one(q_rewards, dq_rewards, q_table, policy_grid, q_stability, dq_stability,
                    title="RL Fighting Project - Complete Analysis"):
    """
    Generate a single figure containing all required plots:
    - Learning curves (Q-learning & Double Q-learning)
    - Q-value heatmap (from Q-learning)
    - Policy visualization (argmax Q)
    - Stability plots (both algorithms)
    - Algorithm comparison (smoothed)
    
    Parameters:
        q_rewards: list of episode rewards for Q-learning
        dq_rewards: list for Double Q-learning
        q_table: numpy array (101,101,3) from QLearningAgent
        policy_grid: 2D array of actions (0,1,2) from argmax of q_table
        q_stability: 2D array (n_trials, episodes) for Q-learning
        dq_stability: 2D array (n_trials, episodes) for Double Q-learning
        title: overall figure title
    """
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # 1. Learning curves (subplot 2,2,1)
    ax1 = plt.subplot(2, 3, 1)
    window = min(30, len(q_rewards)//5)
    if window > 1:
        q_smooth = np.convolve(q_rewards, np.ones(window)/window, mode='valid')
        dq_smooth = np.convolve(dq_rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(q_rewards)), q_smooth, 'b-', label='Q-learning (smoothed)')
        ax1.plot(range(window-1, len(dq_rewards)), dq_smooth, 'r-', label='Double Q-learning (smoothed)')
    else:
        ax1.plot(q_rewards, 'b-', alpha=0.5, label='Q-learning')
        ax1.plot(dq_rewards, 'r-', alpha=0.5, label='Double Q-learning')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Learning Curves')
    ax1.legend()
    ax1.grid(True)
    
    # 2. Q-value heatmap (subplot 2,3,2)
    ax2 = plt.subplot(2, 3, 2)
    opponent_hp = np.arange(0, 101)
    agent_hp_fixed = 50
    q_vals = q_table[agent_hp_fixed, opponent_hp, :]  # shape (101,3)
    sns.heatmap(q_vals.T, cmap='viridis', ax=ax2, cbar=True,
                xticklabels=10, yticklabels=['Hand','Foot','Weapon'])
    ax2.set_xlabel('Opponent HP')
    ax2.set_ylabel('Action')
    ax2.set_title(f'Q-values (Agent HP={agent_hp_fixed})')
    
    # 3. Policy visualization (subplot 2,3,3)
    ax3 = plt.subplot(2, 3, 3)
    im = ax3.imshow(policy_grid, origin='lower', cmap='tab10', interpolation='none')
    cbar = plt.colorbar(im, ax=ax3, ticks=[0,1,2])
    cbar.ax.set_yticklabels(['Hand','Foot','Weapon'])
    ax3.set_xlabel('Opponent HP')
    ax3.set_ylabel('Agent HP')
    ax3.set_title('Learned Policy (argmax Q)')
    
    # 4. Stability – Q-learning (subplot 2,3,4)
    ax4 = plt.subplot(2, 3, 4)
    mean_q = np.mean(q_stability, axis=0)
    std_q = np.std(q_stability, axis=0)
    episodes = range(1, len(mean_q)+1)
    ax4.plot(episodes, mean_q, 'b-', label='Mean reward')
    ax4.fill_between(episodes, mean_q - std_q, mean_q + std_q, alpha=0.3, label='±1 std')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Total Reward')
    ax4.set_title('Stability: Q-learning')
    ax4.legend()
    ax4.grid(True)
    
    # 5. Stability – Double Q-learning (subplot 2,3,5)
    ax5 = plt.subplot(2, 3, 5)
    mean_dq = np.mean(dq_stability, axis=0)
    std_dq = np.std(dq_stability, axis=0)
    ax5.plot(episodes, mean_dq, 'r-', label='Mean reward')
    ax5.fill_between(episodes, mean_dq - std_dq, mean_dq + std_dq, alpha=0.3, label='±1 std')
    ax5.set_xlabel('Episode')
    ax5.set_ylabel('Total Reward')
    ax5.set_title('Stability: Double Q-learning')
    ax5.legend()
    ax5.grid(True)
    
    # 6. Algorithm comparison (already shown in learning curve, but separate for clarity)
    # We'll reuse learning curve data but with raw comparison (subplot 2,3,6)
    ax6 = plt.subplot(2, 3, 6)
    ax6.plot(q_rewards, 'b-', alpha=0.3, label='Q-learning raw')
    ax6.plot(dq_rewards, 'r-', alpha=0.3, label='Double Q-learning raw')
    if window > 1:
        ax6.plot(range(window-1, len(q_rewards)), q_smooth, 'b-', linewidth=2, label='Q-learning smooth')
        ax6.plot(range(window-1, len(dq_rewards)), dq_smooth, 'r-', linewidth=2, label='Double Q-learning smooth')
    ax6.set_xlabel('Episode')
    ax6.set_ylabel('Total Reward')
    ax6.set_title('Final Comparison (raw + smooth)')
    ax6.legend()
    ax6.grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for suptitle
    plt.show()
    return fig


# ------------------------- Individual plot functions (kept for flexibility) -------------------------
def plot_learning_curve(rewards, title="Learning Curve"):
    plt.figure(figsize=(10,5))
    plt.plot(rewards, alpha=0.3, label='Raw')
    window = min(30, len(rewards)//5)
    if window > 1:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards)), moving_avg, 'r-', label=f'Moving avg (w={window})')
    plt.xlabel('Episode'); plt.ylabel('Total Reward'); plt.title(title)
    plt.legend(); plt.grid(True); plt.show()

def plot_q_heatmap(q_table, agent_hp=50, title="Q-value Heatmap"):
    opponent_hp = np.arange(0, 101)
    q_vals = q_table[agent_hp, opponent_hp, :]
    plt.figure(figsize=(10,6))
    sns.heatmap(q_vals.T, cmap='viridis', xticklabels=10, yticklabels=['Hand','Foot','Weapon'])
    plt.xlabel('Opponent HP'); plt.ylabel('Action'); plt.title(f"{title} (Agent HP={agent_hp})")
    plt.tight_layout(); plt.show()

def plot_policy(policy_grid, title="Learned Policy"):
    plt.figure(figsize=(10,8))
    im = plt.imshow(policy_grid, origin='lower', cmap='tab10', interpolation='none')
    cbar = plt.colorbar(im, ticks=[0,1,2])
    cbar.ax.set_yticklabels(['Hand','Foot','Weapon'])
    plt.xlabel('Opponent HP'); plt.ylabel('Agent HP'); plt.title(title)
    plt.tight_layout(); plt.show()

def plot_stability(all_rewards, algorithm_name="Algorithm"):
    mean_r = np.mean(all_rewards, axis=0)
    std_r = np.std(all_rewards, axis=0)
    episodes = range(1, len(mean_r)+1)
    plt.figure(figsize=(10,5))
    plt.plot(episodes, mean_r, 'b-', label='Mean')
    plt.fill_between(episodes, mean_r - std_r, mean_r + std_r, alpha=0.3)
    plt.xlabel('Episode'); plt.ylabel('Total Reward'); plt.title(f'Stability: {algorithm_name}')
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

def compare_algorithms(results_dict):
    plt.figure(figsize=(10,5))
    for name, rewards in results_dict.items():
        window = min(30, len(rewards)//5)
        if window > 1:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(rewards)), moving_avg, label=f'{name} (smoothed)')
        else:
            plt.plot(rewards, label=name)
    plt.xlabel('Episode'); plt.ylabel('Total Reward'); plt.title('Algorithm Comparison')
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()