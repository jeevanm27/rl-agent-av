import tensorflow as tf
import numpy as np
import gymnasium as gym
import highway_env
from collections import deque, defaultdict
import random
import time
from typing import Dict, List, Tuple
import os
import json
import pickle
import shutil

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

class AdaptiveSafetyModule:
    def __init__(self):
        # Core safety parameters
        self.safe_distance = 20.0
        self.safe_speed = 30.0
        self.min_speed = 20.0
        
        # Weights for reward components
        self.collision_weight = 0.4
        self.speed_weight = 0.3
        self.lane_change_weight = 0.3
        
        # Lane change parameters
        self.last_lane_change_time = 0
        self.lane_change_cooldown = 2.0  # seconds
        
    def compute_safety_reward(self, 
                            observation: np.ndarray,
                            action: int,
                            info: dict,
                            time_elapsed: float) -> float:
        """
        Compute safety reward based on actual environment observations.
        observation shape: (vehicles_count, features) where features are [presence, x, y, vx, vy]
        """
        try:
            # Extract ego vehicle state (first vehicle in observation)
            ego_presence, ego_x, ego_y, ego_vx, ego_vy = observation[0]
            ego_speed = np.sqrt(ego_vx**2 + ego_vy**2)
            
            # Initialize rewards
            collision_reward = 0.0
            speed_reward = 0.0
            lane_change_reward = 0.0
            
            # Check other vehicles for potential collisions
            for i in range(1, len(observation)):
                if observation[i][0]:  # if vehicle is present
                    _, other_x, other_y, other_vx, other_vy = observation[i]
                    
                    # Compute relative distance
                    rel_x = other_x - ego_x
                    rel_y = other_y - ego_y
                    distance = np.sqrt(rel_x**2 + rel_y**2)
                    
                    # Collision reward (negative for being too close)
                    if distance < self.safe_distance:
                        collision_reward -= (1.0 - distance/self.safe_distance)
            
            # Speed reward
            if ego_speed < self.min_speed:
                speed_reward = ego_speed/self.min_speed - 1.0
            elif ego_speed > self.safe_speed:
                speed_reward = 1.0 - (ego_speed - self.safe_speed)/self.safe_speed
            else:
                speed_reward = ego_speed/self.safe_speed
            
            # Lane change reward
            if action in [0, 2]:  # Lane change actions
                time_since_change = time_elapsed - self.last_lane_change_time
                if time_since_change > self.lane_change_cooldown:
                    lane_change_reward = 0.5  # Reward for safe lane change
                    self.last_lane_change_time = time_elapsed
                else:
                    lane_change_reward = -0.5  # Penalty for too frequent lane changes
            
            # Combine rewards
            total_reward = (
                self.collision_weight * collision_reward +
                self.speed_weight * speed_reward +
                self.lane_change_weight * lane_change_reward
            )
            
            return total_reward
            
        except Exception as e:
            print(f"Warning: Error in compute_safety_reward: {str(e)}")
            return 0.0  # Safe default value

class ReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4, beta_increment: float = 0.001):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling exponent
        self.beta_increment = beta_increment  # Beta increment per sampling
        self.max_priority = 1.0
        self.tree_idx = 0
        self.size = 0
        
        # Initialize sum tree for priorities
        self.sum_tree = np.zeros(2 * capacity - 1)  # Tree structure
        self.data = np.zeros(capacity, dtype=object)  # Data storage
        self.priorities = np.zeros(capacity)  # Store priorities for updating
        
    def _propagate(self, idx: int, change: float):
        """Propagate priority change up through the tree."""
        parent = (idx - 1) // 2
        self.sum_tree[parent] += change
        if parent > 0:
            self._propagate(parent, change)
            
    def _retrieve(self, idx: int, s: float) -> int:
        """Find sample index given a priority value."""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.sum_tree):
            return idx
            
        if s <= self.sum_tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.sum_tree[left])
            
    def add(self, state, action, reward, next_state, done, error=None):
        """Add new experience with priority."""
        # Use max priority for new experiences
        priority = self.max_priority if error is None else min(abs(error) + 1e-5, self.max_priority)
        
        idx = self.tree_idx
        self.data[idx] = (state, action, reward, next_state, done)
        self.update(idx, priority)
        
        self.tree_idx = (self.tree_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def update(self, idx: int, priority: float):
        """Update priority of experience."""
        priority = min(abs(priority) + 1e-5, self.max_priority) ** self.alpha
        change = priority - self.sum_tree[idx + self.capacity - 1]
        self.sum_tree[idx + self.capacity - 1] = priority
        self.priorities[idx] = priority
        self._propagate(idx + self.capacity - 1, change)
        
    def sample(self, batch_size: int) -> Tuple:
        """Sample batch with priorities."""
        batch_indices = []
        batch = []
        weights = np.zeros(batch_size)
        
        # Compute segment size
        segment = self.sum_tree[0] / batch_size
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            
            idx = self._retrieve(0, s)
            idx_data = idx - self.capacity + 1
            
            if idx_data < 0 or self.data[idx_data] is None:
                # If invalid index or empty data, resample
                idx_data = random.randint(0, max(0, self.size - 1))
            
            batch_indices.append(idx_data)
            batch.append(self.data[idx_data])
        
        # Compute importance sampling weights
        p_min = np.min(self.priorities[:self.size]) / self.sum_tree[0]
        max_weight = (p_min * self.size) ** (-self.beta)
        
        for i, idx in enumerate(batch_indices):
            p_sample = self.sum_tree[idx + self.capacity - 1] / self.sum_tree[0]
            weights[i] = ((p_sample * self.size) ** (-self.beta)) / max_weight
        
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones), weights, batch_indices)
    
    def update_batch_priorities(self, indices: List[int], priorities: List[float]):
        """Update priorities for a batch of experiences."""
        for idx, priority in zip(indices, priorities):
            self.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)
            
    def __len__(self):
        return self.size

class DQNNetwork(tf.keras.Model):
    def __init__(self, state_dim: Tuple[int, ...], action_dim: int, hidden_dims: List[int]):
        super(DQNNetwork, self).__init__()
        
        # Flatten layer to handle 2D state input
        self.flatten = tf.keras.layers.Flatten()
        
        self.layers_list = []
        for dim in hidden_dims:
            self.layers_list.extend([
                tf.keras.layers.Dense(
                    dim, 
                    activation='relu',
                    kernel_initializer=tf.keras.initializers.HeUniform(),
                    bias_initializer=tf.keras.initializers.Zeros()
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.1)
            ])
            
        # Final layer for Q-values with proper initialization
        self.q_head = tf.keras.layers.Dense(
            action_dim,
            kernel_initializer=tf.keras.initializers.HeUniform(),
            bias_initializer=tf.keras.initializers.Zeros()
        )
        
    def call(self, state, training=False):
        x = self.flatten(state)
        for layer in self.layers_list:
            if isinstance(layer, tf.keras.layers.BatchNormalization) or \
               isinstance(layer, tf.keras.layers.Dropout):
                x = layer(x, training=training)
            else:
                x = layer(x)
        return self.q_head(x)

class DQN:
    def __init__(
        self,
        state_dim: Tuple[int, ...],
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.02,
        epsilon_decay_steps: int = 10000,
        tau: float = 0.005,
        buffer_capacity: int = 20000,
        batch_size: int = 128,
        update_target_every: int = 100,
        checkpoint_dir: str = "./checkpoints",
        per_alpha: float = 0.6,
        per_beta: float = 0.4,
        per_beta_increment: float = 0.001
    ):
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps
        self.tau = tau
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.update_target_every = update_target_every
        self.checkpoint_dir = checkpoint_dir
        self.train_step_counter = 0
        self.total_steps = 0
        
        # Initialize networks with proper initialization
        self.q_network = DQNNetwork(state_dim, action_dim, hidden_dims)
        self.target_network = DQNNetwork(state_dim, action_dim, hidden_dims)
        
        # Build models by calling them once
        dummy_state = tf.zeros((1,) + state_dim)
        self.q_network(dummy_state)
        self.target_network(dummy_state)
        
        # Copy weights to target network
        self.target_network.set_weights(self.q_network.get_weights())
        
        # Initialize optimizer with gradient clipping
        self.optimizer = tf.keras.optimizers.Adam(learning_rate, clipnorm=1.0)
        
        # Initialize prioritized replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=buffer_capacity,
            alpha=per_alpha,
            beta=per_beta,
            beta_increment=per_beta_increment
        )
        
        # Initialize Huber loss for more stable training
        self.loss_fn = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)
        
    def _update_target(self):
        """Soft update target network weights."""
        for target_weights, q_weights in zip(self.target_network.weights, self.q_network.weights):
            target_weights.assign(self.tau * q_weights + (1 - self.tau) * target_weights)
            
    def update(self, batch_size: int) -> Dict[str, float]:
        if len(self.replay_buffer) < batch_size:
            return {}
            
        states, actions, rewards, next_states, dones, weights, indices = self.replay_buffer.sample(batch_size)
        
        # Convert to tensors and ensure proper shapes
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        weights = tf.convert_to_tensor(weights, dtype=tf.float32)
        
        # Add batch dimension if necessary
        if len(states.shape) == 1:
            states = tf.expand_dims(states, 0)
            next_states = tf.expand_dims(next_states, 0)
            rewards = tf.expand_dims(rewards, 0)
            dones = tf.expand_dims(dones, 0)
            actions = tf.expand_dims(actions, 0)
            weights = tf.expand_dims(weights, 0)
        
        with tf.GradientTape() as tape:
            # Compute Q values for current states
            q_values = self.q_network(states)
            q_values_selected = tf.gather(q_values, actions, batch_dims=1)
            
            # Compute target Q values with Double DQN
            next_q_values = self.q_network(next_states)
            next_actions = tf.argmax(next_q_values, axis=1)
            next_q_values_target = self.target_network(next_states)
            next_q_values_selected = tf.gather(next_q_values_target, next_actions, batch_dims=1)
            
            # Compute targets with proper broadcasting
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values_selected
            
            # Compute TD errors for updating priorities
            td_errors = target_q_values - q_values_selected
            
            # Compute loss with importance sampling weights
            losses = self.loss_fn(target_q_values, q_values_selected)
            weighted_losses = tf.multiply(losses, weights)
            loss = tf.reduce_mean(weighted_losses)
        
        # Compute and apply gradients with clipping
        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        gradients = [tf.clip_by_norm(g, 1.0) for g in gradients]
        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))
        
        # Update priorities in replay buffer
        td_errors_np = tf.abs(td_errors).numpy()
        self.replay_buffer.update_batch_priorities(indices, td_errors_np)
        
        # Periodically update target network
        self.train_step_counter += 1
        if self.train_step_counter % self.update_target_every == 0:
            self._update_target()
        
        # Update total steps and decay epsilon
        self.total_steps += 1
        if self.total_steps < self.epsilon_decay_steps:
            self.epsilon = max(self.epsilon_end, 
                             self.epsilon - self.epsilon_decay)
        
        return {
            'loss': float(loss),
            'epsilon': float(self.epsilon),
            'max_q': float(tf.reduce_max(q_values)),
            'mean_td_error': float(tf.reduce_mean(tf.abs(td_errors)))
        }
        
    def select_action(self, state, evaluate=False):
        if not evaluate and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
            
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        q_values = self.q_network(state)[0]
        return int(tf.argmax(q_values))
            
    def save_models(self, path: str, save_buffer: bool = True, episode: int = None, metrics: Dict = None):
        """Save model weights, replay buffer, and training metrics."""
        if not os.path.exists(path):
            os.makedirs(path)
            
        # Save model weights
        self.q_network.save_weights(f"{path}/q_network.weights.h5")
        self.target_network.save_weights(f"{path}/target_network.weights.h5")
        
        # Save replay buffer if requested
        if save_buffer:
            buffer_path = os.path.join(path, "replay_buffer.pkl")
            with open(buffer_path, 'wb') as f:
                pickle.dump(self.replay_buffer, f)
        
        # Save training metrics and episode number if provided
        if episode is not None or metrics is not None:
            metadata = {
                'episode': episode,
                'metrics': metrics or {},
                'epsilon': self.epsilon
            }
            with open(os.path.join(path, "training_metadata.json"), 'w') as f:
                json.dump(metadata, f)
        
    def load_models(self, path: str, load_buffer: bool = True):
        """Load model weights and optionally replay buffer."""
        # Load model weights
        self.q_network.load_weights(f"{path}/q_network.weights.h5")
        self.target_network.load_weights(f"{path}/target_network.weights.h5")
        
        # Load replay buffer if requested and exists
        if load_buffer and os.path.exists(os.path.join(path, "replay_buffer.pkl")):
            with open(os.path.join(path, "replay_buffer.pkl"), 'rb') as f:
                self.replay_buffer = pickle.load(f)
                
        # Load and return training metadata if exists
        metadata_path = os.path.join(path, "training_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.epsilon = metadata.get('epsilon', self.epsilon)
                return metadata
        return None

def train_agent(checkpoint_dir="/content/drive/MyDrive/highway_rl_checkpoints", start_episode=0, max_episodes=10000):
    """Train agent with enhanced safety and lane changing behavior."""
    # Ensure checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize safety module
    safety_module = AdaptiveSafetyModule()
    
    # Create and configure the environment with enhanced settings
    env = gym.make('highway-v0', render_mode=None, config={
        'observation': {
            'type': 'Kinematics',
            'features': ['presence', 'x', 'y', 'vx', 'vy'],
            'absolute': True,
            'normalize': True,
            'vehicles_count': 15,
            'features_range': {
                'x': [-100, 100],
                'y': [-100, 100],
                'vx': [-30, 30],
                'vy': [-30, 30]
            },
        },
        'action': {
            'type': 'DiscreteMetaAction',
            'target_speeds': [20, 25, 30],
            'longitudinal': True,
            'lateral': True,
        },
        'lanes_count': 4,
        'vehicles_count': 40,
        'duration': 40,
        'initial_spacing': 2.5,
        'collision_reward': -10.0,  # Increased collision penalty significantly
        'high_speed_reward': 0.4,
        'right_lane_reward': 0.0,   # Removed right lane bias
        'lane_change_reward': 0.4,   # Increased reward for lane changes
        'reward_speed_range': [20, 30],
        'normalize_reward': True,
        'offroad_terminal': True,
        'simulation_frequency': 15,
        'policy_frequency': 5,
        'screen_width': 600,
        'screen_height': 150,
        'centering_position': [0.3, 0.5],
        'scaling': 5.5,
        'show_trajectories': True
    })
    
    # Get state and action dimensions
    state_dim = env.observation_space.shape
    action_dim = env.action_space.n
    
    # Calculate total training steps for epsilon decay
    total_training_steps = max_episodes * 1000  # max_steps per episode
    
    # Initialize agent with optimized hyperparameters for DQN
    agent = DQN(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[256, 256],
        learning_rate=2.5e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.02,
        epsilon_decay_steps=10000,  # Decay over 10000 steps
        tau=0.005,
        buffer_capacity=20000,  # Reduced from 100000 to 20000 (about 20 episodes worth of experience)
        batch_size=128,
        update_target_every=100,
        checkpoint_dir=checkpoint_dir
    )
    
    # Try to load latest checkpoint
    latest_checkpoint = None
    if os.path.exists(checkpoint_dir):
        checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint_")]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("_")[1]))
            checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
            metadata = agent.load_models(checkpoint_path, load_buffer=True)
            if metadata:
                start_episode = metadata.get('episode', 0) + 1
                print(f"Resuming training from episode {start_episode}")
    
    # Training parameters
    max_steps = 1000
    batch_size = 64
    eval_interval = 10
    checkpoint_interval = 50
    min_samples = 1000
    
    # Training metrics
    episode_rewards = []
    training_metrics = {
        'episode_rewards': [],
        'avg_rewards': [],
        'losses': [],
        'epsilons': [],
        'max_q_values': []
    }
    
    # Initialize running metrics
    running_loss = []
    running_max_q = []
    
    print(f"Starting training from episode {start_episode}")
    print(f"Checkpoints will be saved every {checkpoint_interval} episodes to: {checkpoint_dir}")
    
    try:
        for episode in range(start_episode, max_episodes):
            state, _ = env.reset()
            episode_reward = 0
            episode_start_time = time.time()
            episode_losses = []
            episode_max_qs = []
            
            for step in range(max_steps):
                action = agent.select_action(state)
                next_state, reward, done, truncated, info = env.step(action)
                
                # Get current time for safety module
                time_elapsed = time.time() - episode_start_time
                
                # Compute safety reward
                safety_reward = safety_module.compute_safety_reward(
                    state, action, info, time_elapsed
                )
                
                # Combine rewards
                combined_reward = (
                    0.4 * reward +  # Original environment reward
                    0.6 * safety_reward  # Safety considerations
                )
                
                # Clip combined reward for stability
                combined_reward = np.clip(combined_reward, -10.0, 2.0)
                
                # Store transition
                agent.replay_buffer.add(
                    state=state,
                    action=action,
                    reward=combined_reward,
                    next_state=next_state,
                    done=done,
                    error=None
                )
                
                # Update network if we have enough samples
                if len(agent.replay_buffer) > min_samples:
                    update_info = agent.update(batch_size)
                    if update_info:
                        episode_losses.append(update_info['loss'])
                        episode_max_qs.append(update_info['max_q'])
                        
                        # Track TD errors for monitoring learning progress
                        if 'mean_td_error' in update_info:
                            print(f"Step {step}, Mean TD Error: {update_info['mean_td_error']:.3f}", end='\r')
                
                episode_reward += combined_reward
                state = next_state
                
                if done or truncated:
                    break
            
            # Store episode metrics
            episode_rewards.append(episode_reward)
            training_metrics['episode_rewards'].append(episode_reward)
            
            # Update running metrics
            if episode_losses:
                avg_loss = np.mean(episode_losses)
                running_loss.append(avg_loss)
                training_metrics['losses'].append(avg_loss)
            
            if episode_max_qs:
                avg_max_q = np.mean(episode_max_qs)
                running_max_q.append(avg_max_q)
                training_metrics['max_q_values'].append(avg_max_q)
            
            training_metrics['epsilons'].append(agent.epsilon)
            
            # Print progress
            if (episode + 1) % eval_interval == 0:
                avg_reward = np.mean(episode_rewards[-eval_interval:])
                training_metrics['avg_rewards'].append(avg_reward)
                
                # Compute metrics only if we have data
                avg_loss = np.mean(running_loss[-eval_interval:]) if running_loss else float('nan')
                avg_max_q = np.mean(running_max_q[-eval_interval:]) if running_max_q else float('nan')
                
                print(f"Episode {episode + 1}, "
                      f"Average Reward: {avg_reward:.2f}, "
                      f"Epsilon: {agent.epsilon:.3f}, "
                      f"Loss: {avg_loss:.3f}, "
                      f"Max Q: {avg_max_q:.3f}")
            
            # Save checkpoint
            if (episode + 1) % checkpoint_interval == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{episode + 1}")
                
                # Save with temporary name first
                temp_checkpoint_path = os.path.join(checkpoint_dir, f"temp_checkpoint_{episode + 1}")
                agent.save_models(
                    temp_checkpoint_path,
                    save_buffer=True,
                    episode=episode,
                    metrics=training_metrics
                )
                
                # Safely replace old checkpoint
                if os.path.exists(checkpoint_path):
                    shutil.rmtree(checkpoint_path)
                os.rename(temp_checkpoint_path, checkpoint_path)
                
                # Keep only the latest checkpoint
                for old_checkpoint in os.listdir(checkpoint_dir):
                    if old_checkpoint.startswith("checkpoint_") and old_checkpoint != f"checkpoint_{episode + 1}":
                        old_path = os.path.join(checkpoint_dir, old_checkpoint)
                        if os.path.exists(old_path):
                            shutil.rmtree(old_path)
                
                print(f"Checkpoint saved at episode {episode + 1}")
                
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving final checkpoint...")
        final_checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_interrupted_{episode}")
        agent.save_models(
            final_checkpoint_path,
            save_buffer=True,
            episode=episode,
            metrics=training_metrics
        )
        print(f"Interrupted checkpoint saved at: {final_checkpoint_path}")
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        print("Attempting to save emergency checkpoint...")
        emergency_path = os.path.join(checkpoint_dir, "emergency_checkpoint")
        try:
            agent.save_models(
                emergency_path,
                save_buffer=True,
                episode=episode,
                metrics={
                    'episode_rewards': episode_rewards,
                    'error': str(e)
                }
            )
            print(f"Emergency checkpoint saved at: {emergency_path}")
        except:
            print("Failed to save emergency checkpoint")
        raise
        
    return agent, episode_rewards

def evaluate_agent(agent, num_episodes=10):
    env = gym.make('highway-v0', render_mode=None, config={
        'observation': {
            'type': 'Kinematics',
            'features': ['presence', 'x', 'y', 'vx', 'vy'],
            'absolute': True,
            'normalize': True,
            'vehicles_count': 15,  # Match the training environment
            'features_range': {
                'x': [-100, 100],
                'y': [-100, 100],
                'vx': [-30, 30],
                'vy': [-30, 30]
            },
        },
        'action': {
            'type': 'DiscreteMetaAction',
            'target_speeds': [20, 25, 30],  # Match the training environment
            'longitudinal': True,
            'lateral': True,
        },
        'lanes_count': 4,
        'vehicles_count': 40,  # Match the training environment
        'duration': 40,
        'initial_spacing': 2.5
    })
    
    eval_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state, evaluate=True)
            next_state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            state = next_state
            
            if truncated:
                break
                
        eval_rewards.append(episode_reward)
        print(f"Evaluation Episode {episode + 1}, Reward: {episode_reward:.2f}")
        
    return np.mean(eval_rewards)

if __name__ == "__main__":
    # Mount Google Drive for Colab
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("Google Drive mounted successfully")
    except ImportError:
        print("Not running in Colab environment")
    
    # Set up checkpoint directory in Google Drive
    CHECKPOINT_DIR = "/content/drive/MyDrive/highway_rl_checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    print("Starting training...")
    print(f"Checkpoints will be saved to Google Drive at: {CHECKPOINT_DIR}")
    
    # Verify Google Drive access
    try:
        test_file = os.path.join(CHECKPOINT_DIR, "test.txt")
        with open(test_file, "w") as f:
            f.write("Test file to verify Google Drive access")
        os.remove(test_file)
        print("Successfully verified Google Drive write access")
    except Exception as e:
        print(f"Warning: Could not verify Google Drive access. Error: {str(e)}")
        print("Please ensure Google Drive is mounted properly")
    
    agent, training_rewards = train_agent(checkpoint_dir=CHECKPOINT_DIR)
    
    # Evaluate the trained agent
    print("\nStarting evaluation...")
    mean_eval_reward = evaluate_agent(agent)
    print(f"\nMean evaluation reward: {mean_eval_reward:.2f}")
    
    print(f"\nCheckpoints have been saved to: {CHECKPOINT_DIR}")
    print("You can find your saved models in Google Drive under the 'highway_rl_checkpoints' folder") 
