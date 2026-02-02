"""
PPO Agent for Trading.

Proximal Policy Optimization (PPO) implementation for trading decisions.

Based on agent.md Section 3.3:
- Actor-Critic architecture
- Clipped surrogate objective
- Multiple epochs per batch
- Generalized Advantage Estimation (GAE)

Falls back to random actions if PyTorch not available.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import warnings

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Categorical

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    warnings.warn("PyTorch not available. Using random policy fallback.")


@dataclass
class PPOConfig:
    """PPO hyperparameters."""

    learning_rate: float = 3e-4
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda
    epsilon: float = 0.2  # PPO clipping parameter
    value_coef: float = 0.5  # Value loss coefficient
    entropy_coef: float = 0.01  # Entropy bonus coefficient
    max_grad_norm: float = 0.5  # Gradient clipping
    update_epochs: int = 4  # Epochs per update
    batch_size: int = 64  # Mini-batch size
    hidden_dim: int = 128  # Hidden layer dimension


class PPONetwork(nn.Module if HAS_TORCH else object):
    """
    Actor-Critic network for PPO.

    Architecture:
    - Shared feature extractor
    - Actor head (policy)
    - Critic head (value)
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        """
        Initialize the network.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Hidden layer dimension
        """
        if not HAS_TORCH:
            self.state_dim = state_dim
            self.action_dim = action_dim
            return

        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Actor (policy) head
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=-1),
        )

        # Critic (value) head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)

    def forward(self, state: Any) -> Tuple[Any, Any]:
        """
        Forward pass.

        Args:
            state: State tensor

        Returns:
            action_probs: Action probability distribution
            value: State value estimate
        """
        if not HAS_TORCH:
            return None, None

        features = self.shared(state)
        action_probs = self.actor(features)
        value = self.critic(features)

        return action_probs, value

    def get_action(
        self, state: np.ndarray, training: bool = True
    ) -> Tuple[int, float, float]:
        """
        Get action from policy.

        Args:
            state: State observation
            training: Whether in training mode (sample) or eval (argmax)

        Returns:
            action: Selected action
            log_prob: Log probability of action
            value: Value estimate
        """
        if not HAS_TORCH:
            # Random policy fallback
            action = np.random.randint(0, self.action_dim)
            return action, 0.0, 0.0

        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            action_probs, value = self.forward(state_tensor)

        if training:
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        else:
            action = torch.argmax(action_probs, dim=-1)
            log_prob = torch.log(action_probs[0, action])

        return action.item(), log_prob.item(), value.item()


class PPOAgent:
    """
    Proximal Policy Optimization agent for trading.

    Advantages:
    - Stable training with clipped objective
    - Sample efficient (multiple epochs per batch)
    - Works well with discrete actions

    Based on agent.md Section 3.3.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Optional[PPOConfig] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize PPO agent.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            config: PPO configuration
            device: 'cuda' or 'cpu'
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or PPOConfig()

        if HAS_TORCH:
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

            # Initialize network
            self.network = PPONetwork(state_dim, action_dim, self.config.hidden_dim).to(
                self.device
            )

            # Optimizer
            self.optimizer = optim.Adam(
                self.network.parameters(), lr=self.config.learning_rate
            )
        else:
            self.device = "cpu"
            self.network = PPONetwork(state_dim, action_dim)
            self.optimizer = None

        # Experience buffer
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.log_probs: List[float] = []
        self.dones: List[bool] = []

        # Training stats
        self.training_stats: List[Dict] = []

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using current policy.

        Args:
            state: Current state
            training: Whether in training mode

        Returns:
            Selected action
        """
        action, log_prob, value = self.network.get_action(state, training)

        if training:
            self.states.append(state)
            self.actions.append(action)
            self.values.append(value)
            self.log_probs.append(log_prob)

        return action

    def store_transition(self, reward: float, done: bool):
        """Store reward and done flag for current transition."""
        self.rewards.append(reward)
        self.dones.append(done)

    def compute_gae(self, next_value: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation.

        Args:
            next_value: Value estimate of next state

        Returns:
            advantages: GAE advantages
            returns: Discounted returns
        """
        advantages = []
        returns = []
        gae = 0

        values = self.values + [next_value]

        for t in reversed(range(len(self.rewards))):
            if self.dones[t]:
                delta = self.rewards[t] - values[t]
                gae = delta
            else:
                delta = self.rewards[t] + self.config.gamma * values[t + 1] - values[t]
                gae = delta + self.config.gamma * self.config.gae_lambda * gae

            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])

        advantages = np.array(advantages)
        returns = np.array(returns)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def update(self, next_state: np.ndarray) -> Dict:
        """
        Update policy using PPO.

        Args:
            next_state: State after last action

        Returns:
            Training statistics
        """
        if not HAS_TORCH:
            self._clear_buffer()
            return {"note": "No training without PyTorch"}

        if len(self.states) == 0:
            return {"note": "Empty buffer"}

        # Get next value
        with torch.no_grad():
            state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            _, next_value = self.network(state_tensor)
            next_value = next_value.item()

        # Compute GAE
        advantages, returns = self.compute_gae(next_value)

        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)

        # PPO update (multiple epochs)
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        n_updates = 0

        n_samples = len(states)
        indices = np.arange(n_samples)

        for _ in range(self.config.update_epochs):
            np.random.shuffle(indices)

            for start in range(0, n_samples, self.config.batch_size):
                end = min(start + self.config.batch_size, n_samples)
                batch_indices = indices[start:end]

                # Get batch
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]

                # Forward pass
                action_probs, values = self.network(batch_states)
                values = values.squeeze()

                # New log probs and entropy
                dist = Categorical(action_probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # PPO loss
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.config.epsilon, 1 + self.config.epsilon)
                    * batch_advantages
                )

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(values, batch_returns)

                # Total loss
                loss = (
                    actor_loss
                    + self.config.value_coef * critic_loss
                    - self.config.entropy_coef * entropy
                )

                # Update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

        # Clear buffer
        self._clear_buffer()

        stats = {
            "actor_loss": total_actor_loss / n_updates,
            "critic_loss": total_critic_loss / n_updates,
            "entropy": total_entropy / n_updates,
            "n_samples": n_samples,
        }

        self.training_stats.append(stats)

        return stats

    def _clear_buffer(self):
        """Clear experience buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def train(
        self,
        env: Any,
        n_episodes: int = 1000,
        update_frequency: int = 2048,
        verbose: int = 1,
        eval_frequency: int = 100,
    ) -> Dict:
        """
        Train agent on environment.

        Args:
            env: Trading environment
            n_episodes: Number of episodes to train
            update_frequency: Steps between policy updates
            verbose: Verbosity level
            eval_frequency: Episodes between evaluations

        Returns:
            Training history
        """
        episode_rewards = []
        episode_lengths = []
        best_reward = float("-inf")

        total_steps = 0

        for episode in range(n_episodes):
            state = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False

            while not done:
                # Select action
                action = self.select_action(state, training=True)

                # Take step
                next_state, reward, done, info = env.step(action)

                # Store transition
                self.store_transition(reward, done)

                episode_reward += reward
                episode_length += 1
                total_steps += 1

                state = next_state

                # Update policy
                if total_steps % update_frequency == 0 and len(self.states) > 0:
                    update_stats = self.update(next_state)

                    if verbose >= 2:
                        print(
                            f"  Update: actor_loss={update_stats.get('actor_loss', 0):.4f}, "
                            f"critic_loss={update_stats.get('critic_loss', 0):.4f}"
                        )

            # Final update at episode end
            if len(self.states) > 0:
                self.update(state)

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            # Track best
            if episode_reward > best_reward:
                best_reward = episode_reward

            # Logging
            if verbose >= 1 and (episode + 1) % eval_frequency == 0:
                avg_reward = np.mean(episode_rewards[-eval_frequency:])
                avg_length = np.mean(episode_lengths[-eval_frequency:])

                metrics = env.get_metrics()

                print(f"Episode {episode + 1}/{n_episodes}")
                print(f"  Avg Reward: {avg_reward:.2f}")
                print(f"  Avg Length: {avg_length:.0f}")
                print(f"  Total Return: {metrics.get('total_return', 0):.2%}")
                print(f"  Sharpe: {metrics.get('sharpe_ratio', 0):.2f}")

        return {
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "best_reward": best_reward,
            "training_stats": self.training_stats,
        }

    def save(self, path: str):
        """Save agent to disk."""
        if not HAS_TORCH:
            warnings.warn("Cannot save without PyTorch")
            return

        torch.save(
            {
                "network_state_dict": self.network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
            },
            path,
        )

    def load(self, path: str):
        """Load agent from disk."""
        if not HAS_TORCH:
            warnings.warn("Cannot load without PyTorch")
            return

        checkpoint = torch.load(path, map_location=self.device)

        self.state_dim = checkpoint["state_dim"]
        self.action_dim = checkpoint["action_dim"]
        self.config = checkpoint["config"]

        self.network = PPONetwork(
            self.state_dim, self.action_dim, self.config.hidden_dim
        ).to(self.device)
        self.network.load_state_dict(checkpoint["network_state_dict"])

        self.optimizer = optim.Adam(
            self.network.parameters(), lr=self.config.learning_rate
        )
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
