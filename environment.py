"""
environment.py
==============
FightingEnv: A turn-based combat Markov Decision Process (MDP) for training
Q-Learning and Double Q-Learning agents.

State Space  : (101, 101) — discretized HP grid (agent_hp // 10, opp_hp // 10)
Action Space : 3 discrete actions — HAND (0), FOOT (1), WEAPON (2)
Reward       : Shaped to encourage fast, aggressive, winning play.

Compatible with: agents.py, training.py, visualization.py, animation.py
"""

import random
import numpy as np


# ---------------------------------------------------------------------------
# Constants — centralised so every module that imports this file can reference
# them without hard-coding magic numbers.
# ---------------------------------------------------------------------------
ACTION_NAMES: dict[int, str] = {0: "HAND", 1: "FOOT", 2: "WEAPON"}
DAMAGE_MAP:   dict[int, int] = {0: 10,    1: 15,    2: 25}

# Reward shaping coefficients (kept as module-level constants for easy tuning)
_HIT_MULTIPLIER    =  5     # reward per point of damage dealt
_BLOCK_PENALTY     = -20    # penalty when own attack is blocked
_HIT_TAKEN_MULT    = -3     # penalty per point of damage received
_WIN_BONUS         =  1000  # terminal reward for winning
_LOSS_PENALTY      = -1000  # terminal penalty for losing
_TIMEOUT_PENALTY   = -500   # terminal penalty for exceeding max_steps
_STEP_PENALTY      = -1     # constant per-step time pressure


class FightingEnv:
    """Turn-based combat environment compatible with tabular Q-Learning.

    Both combatants begin with ``max_hp`` health points and alternate actions
    each step until one reaches 0 HP or ``max_steps`` is exceeded.

    The state is a 2-tuple of discretised health values so that the Q-table
    indices never exceed the (101, 101) grid expected by the agents.

    Args:
        block_prob (float): Probability [0, 1] that the opponent successfully
            blocks the agent's attack this step. Defaults to 0.2.
        max_hp (int): Starting (and maximum) HP for both combatants.
            Defaults to 1000.
        max_steps (int): Maximum steps per episode before a timeout is issued.
            Prevents unbounded episodes that would stall training. Defaults
            to 200.

    Attributes:
        action_names (dict): Maps action id → human-readable label.
        damage_map (dict): Maps action id → damage value.
        agent_hp (int): Current agent health points.
        opponent_hp (int): Current opponent health points.
        steps (int): Steps taken in the current episode.

    Example:
        >>> env = FightingEnv(block_prob=0.2, max_hp=1000)
        >>> state = env.reset()
        >>> next_state, reward, done, info = env.step(action=2)  # WEAPON
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        block_prob: float = 0.2,
        max_hp: int = 1000,
        max_steps: int = 200,
    ) -> None:
        # Validate inputs early to surface configuration errors immediately.
        if not 0.0 <= block_prob <= 1.0:
            raise ValueError(f"block_prob must be in [0, 1], got {block_prob}")
        if max_hp <= 0:
            raise ValueError(f"max_hp must be positive, got {max_hp}")
        if max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {max_steps}")

        # Configuration
        self.max_hp    = max_hp
        self.block_prob = block_prob
        self.max_steps  = max_steps

        # Expose action metadata so other modules (animation, viz) can import
        # these from the env instance rather than duplicating them.
        self.action_names = ACTION_NAMES
        self.damage_map   = DAMAGE_MAP

        # Runtime state — initialised properly by reset()
        self.agent_hp    = max_hp
        self.opponent_hp = max_hp
        self.steps       = 0

        # Diagnostic attributes consumed by animation.py and visualization.py
        self.last_agent_action:    int | None = None
        self.last_opponent_action: int | None = None
        self.last_block_success:   bool       = False
        self.last_damage_dealt:    int        = 0
        self.last_damage_taken:    int        = 0
        # Alias used by animation.py (keeps backward compatibility)
        self.last_damage:          int        = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> tuple[int, int]:
        """Reset the environment to the start of a new episode.

        Returns:
            tuple[int, int]: Initial discretised state
            ``(agent_hp_scaled, opp_hp_scaled)`` where each value is in
            ``[0, 100]``.
        """
        self.agent_hp    = self.max_hp
        self.opponent_hp = self.max_hp
        self.steps       = 0

        # Clear all diagnostic state
        self.last_agent_action    = None
        self.last_opponent_action = None
        self.last_block_success   = False
        self.last_damage_dealt    = 0
        self.last_damage_taken    = 0
        self.last_damage          = 0

        return self._get_state()

    def step(self, agent_action: int) -> tuple[tuple[int, int], float, bool, dict]:
        """Execute one combat turn and return the RL transition tuple.

        Turn order:
            1. Agent attacks → damage applied if not blocked.
            2. Opponent counter-attacks (only if still alive after step 1).
            3. Terminal conditions are checked; reward components are summed.

        Args:
            agent_action (int): One of {0 (HAND), 1 (FOOT), 2 (WEAPON)}.

        Returns:
            next_state (tuple[int, int]): Discretised ``(agent_hp, opp_hp)``
                after this turn.
            reward (float): Shaped step reward (see module docstring).
            done (bool): ``True`` when the episode has ended.
            info (dict): Diagnostic dictionary with the following keys:

                - ``agent_hp`` (int): Raw agent HP after this step.
                - ``opponent_hp`` (int): Raw opponent HP after this step.
                - ``damage_dealt`` (int): Damage landed on the opponent (0 if
                  blocked).
                - ``damage_taken`` (int): Damage received from the opponent.
                - ``is_blocked`` (bool): Whether the agent's attack was blocked.
                - ``agent_action`` (int): The action taken by the agent.
                - ``opponent_action`` (int | None): The action taken by the
                  opponent (``None`` if opponent was already defeated).
                - ``step_count`` (int): Total steps in the current episode.
                - ``result`` (str): ``"win"``, ``"loss"``, ``"timeout"``, or
                  ``"ongoing"`` — useful for logging win-rates.

        Raises:
            ValueError: If ``agent_action`` is not in ``{0, 1, 2}``.
        """
        if agent_action not in self.damage_map:
            raise ValueError(
                f"Invalid action {agent_action!r}. Must be one of "
                f"{sorted(self.damage_map.keys())}."
            )

        self.last_agent_action = agent_action
        self.steps += 1
        reward = 0.0
        done   = False
        result = "ongoing"

        # ---- 1. Agent's attack ----------------------------------------
        agent_damage = self.damage_map[agent_action]
        blocked      = self._is_blocked()
        self.last_block_success = blocked

        if not blocked:
            self.opponent_hp       = int(np.clip(self.opponent_hp - agent_damage, 0, self.max_hp))
            self.last_damage_dealt = agent_damage
            reward += agent_damage * _HIT_MULTIPLIER
        else:
            self.last_damage_dealt = 0
            reward += _BLOCK_PENALTY

        # Expose alias used by animation.py
        self.last_damage = self.last_damage_dealt

        # ---- 2. Opponent counter-attack (skipped if opponent defeated) -----
        # Guard ensures the opponent cannot land a "revenge hit" after dying.
        if self.opponent_hp > 0:
            opp_action = self._opponent_policy()
            self.last_opponent_action = opp_action

            opp_damage       = self.damage_map[opp_action]
            self.agent_hp    = int(np.clip(self.agent_hp - opp_damage, 0, self.max_hp))
            self.last_damage_taken = opp_damage
            reward += opp_damage * _HIT_TAKEN_MULT
        else:
            # Opponent was knocked out this turn — no counter-attack.
            self.last_opponent_action = None
            self.last_damage_taken    = 0

        # ---- 3. Terminal conditions ------------------------------------
        if self.agent_hp <= 0 and self.opponent_hp <= 0:
            # Simultaneous KO: treat as a loss (rare but possible if both start
            # at very low HP). Designers may prefer a draw — change here.
            reward += _LOSS_PENALTY
            done    = True
            result  = "loss"
        elif self.agent_hp <= 0:
            reward += _LOSS_PENALTY
            done    = True
            result  = "loss"
        elif self.opponent_hp <= 0:
            reward += _WIN_BONUS
            done    = True
            result  = "win"
        elif self.steps >= self.max_steps:
            reward += _TIMEOUT_PENALTY
            done    = True
            result  = "timeout"

        # ---- 4. Per-step time penalty (applied every step) ------------
        reward += _STEP_PENALTY

        # ---- 5. Build info dictionary ---------------------------------
        info: dict = {
            "agent_hp":       self.agent_hp,
            "opponent_hp":    self.opponent_hp,
            "damage_dealt":   self.last_damage_dealt,
            "damage_taken":   self.last_damage_taken,
            "is_blocked":     self.last_block_success,
            "agent_action":   self.last_agent_action,
            "opponent_action": self.last_opponent_action,
            "step_count":     self.steps,
            "result":         result,
        }

        return self._get_state(), reward, done, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_state(self) -> tuple[int, int]:
        """Discretise raw HP into a (101, 101) tabular-Q-compatible state.

        Each HP value is divided by 10 and clipped to [0, 100] so that array
        indices into the Q-table are always valid regardless of floating-point
        rounding or boundary conditions.

        Returns:
            tuple[int, int]: ``(agent_hp_scaled, opp_hp_scaled)`` each in
            ``[0, 100]``.
        """
        scaled_agent = int(np.clip(self.agent_hp // 10, 0, 100))
        scaled_opp   = int(np.clip(self.opponent_hp // 10, 0, 100))
        return (scaled_agent, scaled_opp)

    def _opponent_policy(self) -> int:
        """Select the opponent's action for this turn.

        Currently uses a uniform-random policy so that every action is
        equally likely. The method is intentionally isolated so it can be
        overridden or swapped for a heuristic/rule-based policy without
        touching the rest of the environment logic.

        Returns:
            int: Chosen action in ``{0, 1, 2}``.
        """
        return random.choice(list(self.damage_map.keys()))

    def _is_blocked(self) -> bool:
        """Sample whether the opponent blocks the agent's attack.

        Returns:
            bool: ``True`` with probability ``self.block_prob``.
        """
        return random.random() < self.block_prob

    # ------------------------------------------------------------------
    # Convenience / debugging
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"FightingEnv("
            f"agent_hp={self.agent_hp}/{self.max_hp}, "
            f"opponent_hp={self.opponent_hp}/{self.max_hp}, "
            f"step={self.steps}/{self.max_steps}, "
            f"block_prob={self.block_prob})"
        )