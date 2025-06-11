import tensorflow as tf
import numpy as np
import gymnasium as gym
import highway_env
from collections import deque
import random
import time
from typing import Dict, List, Tuple
import os

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

class ReplayBuffer:
    def __init__(self, capacity: int, recent_size: int, epsilon: float = 0.4):
        self.capacity = capacity
        self.recent_size = recent_size
        self.epsilon = epsilon
        self.buffer = []
        self.priorities = []
        self.position = 0
        
    def add(self, state, action, reward, next_state, done, td_error=None):
        if td_error is None:
            td_error = max(self.priorities) if self.priorities else 1.0
            
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
            self.priorities.append(None)
        
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.priorities[self.position] = td_error
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int) -> Tuple:
        if random.random() < self.epsilon and len(self.buffer) > self.recent_size:
            # Sample from recent transitions
            indices = range(max(0, len(self.buffer) - self.recent_size), len(self.buffer))
        else:
            # Sample based on priorities
            priorities = np.array(self.priorities[:len(self.buffer)])
            probs = priorities / priorities.sum()
            indices = np.random.choice(len(self.buffer), size=batch_size, p=probs)
        
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for idx in indices:
            state, action, reward, next_state, done = self.buffer[idx]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones), indices)
    
    def update_priorities(self, indices: List[int], td_errors: List[float]):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + 1e-6  # Small constant for stability
            
    def __len__(self):
        return len(self.buffer)

class Actor(tf.keras.Model):
    def __init__(self, state_dim: Tuple[int, ...], action_dim: int, hidden_dims: List[int]):
        super(Actor, self).__init__()
        self.action_dim = action_dim
        
        # Flatten layer to handle 2D state input
        self.flatten = tf.keras.layers.Flatten()
        
        self.layers_list = []
        for dim in hidden_dims:
            self.layers_list.extend([
                tf.keras.layers.Dense(dim, activation='relu'),
                tf.keras.layers.LayerNormalization()
            ])
        
        # Final layer for action logits
        self.action_head = tf.keras.layers.Dense(action_dim)
        
    def call(self, state):
        x = self.flatten(state)
        for layer in self.layers_list:
            x = layer(x)
        return self.action_head(x)
    
    def get_action_probs(self, state):
        logits = self(state)  # Shape: [batch_size, action_dim]
        return tf.nn.softmax(logits, axis=-1)  # Shape: [batch_size, action_dim]

class Critic(tf.keras.Model):
    def __init__(self, state_dim: Tuple[int, ...], action_dim: int, hidden_dims: List[int]):
        super(Critic, self).__init__()
        
        # Flatten layer to handle 2D state input
        self.flatten = tf.keras.layers.Flatten()
        
        self.layers_list = []
        for dim in hidden_dims:
            self.layers_list.extend([
                tf.keras.layers.Dense(dim, activation='relu'),
                tf.keras.layers.LayerNormalization()
            ])
            
        # Final layer for Q-values
        self.q_head = tf.keras.layers.Dense(action_dim)
        
    def call(self, state):
        x = self.flatten(state)
        for layer in self.layers_list:
            x = layer(x)
        return self.q_head(x)  # Shape: [batch_size, action_dim]

class DBSAC:
    def __init__(
        self,
        state_dim: Tuple[int, ...],
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        buffer_capacity: int = 100000,
        recent_size: int = 10000,
        batch_size: int = 256
    ):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.action_dim = action_dim
        
        # Initialize networks
        self.actor = Actor(state_dim, action_dim, hidden_dims)
        self.critic_1 = Critic(state_dim, action_dim, hidden_dims)
        self.critic_2 = Critic(state_dim, action_dim, hidden_dims)
        self.target_critic_1 = Critic(state_dim, action_dim, hidden_dims)
        self.target_critic_2 = Critic(state_dim, action_dim, hidden_dims)
        
        # Build models by calling them once
        dummy_state = tf.zeros((1,) + state_dim)
        self.actor(dummy_state)
        self.critic_1(dummy_state)
        self.critic_2(dummy_state)
        self.target_critic_1(dummy_state)
        self.target_critic_2(dummy_state)
        
        # Copy weights to target networks
        self.target_critic_1.set_weights(self.critic_1.get_weights())
        self.target_critic_2.set_weights(self.critic_2.get_weights())
        
        # Initialize optimizers
        self.actor_optimizer = tf.keras.optimizers.Adam(actor_lr)
        self.critic_1_optimizer = tf.keras.optimizers.Adam(critic_lr)
        self.critic_2_optimizer = tf.keras.optimizers.Adam(critic_lr)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity, recent_size)
        
    def _update_target(self, target_weights, weights):
        """Update target weights using exponential moving average."""
        for target_w, w in zip(target_weights, weights):
            target_w.assign(self.tau * w + (1 - self.tau) * target_w)
            
    def update(self, batch_size: int) -> Dict[str, float]:
        if len(self.replay_buffer) < batch_size:
            return {}
            
        states, actions, rewards, next_states, dones, indices = self.replay_buffer.sample(batch_size)
        
        # Convert to tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        
        with tf.GradientTape(persistent=True) as tape:
            # Actor loss
            action_probs = self.actor.get_action_probs(states)
            log_action_probs = tf.math.log(action_probs + 1e-8)
            
            q1_values = self.critic_1(states)
            q2_values = self.critic_2(states)
            min_q_values = tf.minimum(q1_values, q2_values)
            
            actor_loss = tf.reduce_mean(
                tf.reduce_sum(
                    action_probs * (self.alpha * log_action_probs - min_q_values),
                    axis=1
                )
            )
            
            # Critic loss
            next_action_probs = self.actor.get_action_probs(next_states)
            next_log_action_probs = tf.math.log(next_action_probs + 1e-8)
            
            next_q1_values = self.target_critic_1(next_states)
            next_q2_values = self.target_critic_2(next_states)
            next_min_q_values = tf.minimum(next_q1_values, next_q2_values)
            
            next_values = tf.reduce_sum(
                next_action_probs * (next_min_q_values - self.alpha * next_log_action_probs),
                axis=1
            )
            
            target_q_values = rewards + (1 - dones) * self.gamma * next_values
            target_q_values = tf.stop_gradient(target_q_values)
            
            q1_values_selected = tf.reduce_sum(q1_values * tf.one_hot(actions, self.action_dim), axis=1)
            q2_values_selected = tf.reduce_sum(q2_values * tf.one_hot(actions, self.action_dim), axis=1)
            
            critic_1_loss = tf.reduce_mean(tf.square(q1_values_selected - target_q_values))
            critic_2_loss = tf.reduce_mean(tf.square(q2_values_selected - target_q_values))
        
        # Update networks
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        critic_1_grads = tape.gradient(critic_1_loss, self.critic_1.trainable_variables)
        critic_2_grads = tape.gradient(critic_2_loss, self.critic_2.trainable_variables)
        
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        self.critic_1_optimizer.apply_gradients(zip(critic_1_grads, self.critic_1.trainable_variables))
        self.critic_2_optimizer.apply_gradients(zip(critic_2_grads, self.critic_2.trainable_variables))
        
        del tape
        
        # Update target networks
        self._update_target(self.target_critic_1.variables, self.critic_1.variables)
        self._update_target(self.target_critic_2.variables, self.critic_2.variables)
        
        # Update priorities in replay buffer
        td_errors = tf.abs(q1_values_selected - target_q_values)
        self.replay_buffer.update_priorities(indices, td_errors.numpy())
        
        return {
            'actor_loss': float(actor_loss),
            'critic_1_loss': float(critic_1_loss),
            'critic_2_loss': float(critic_2_loss)
        }
        
    def select_action(self, state, evaluate=False):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        action_probs = self.actor.get_action_probs(state)[0]  # Shape: [num_actions]
        
        if evaluate:
            action = tf.argmax(action_probs).numpy()
        else:
            # Reshape to [1, num_actions] for tf.random.categorical
            action_probs = tf.reshape(action_probs, (1, -1))
            action = tf.random.categorical(tf.math.log(action_probs + 1e-8), 1)[0, 0].numpy()
        
        # Ensure action is within bounds
        return int(tf.clip_by_value(action, 0, self.action_dim - 1))
            
    def save_models(self, path: str):
        if not os.path.exists(path):
            os.makedirs(path)
        self.actor.save_weights(f"{path}/actor.weights.h5")
        self.critic_1.save_weights(f"{path}/critic_1.weights.h5")
        self.critic_2.save_weights(f"{path}/critic_2.weights.h5")
        
    def load_models(self, path: str):
        self.actor.load_weights(f"{path}/actor.weights.h5")
        self.critic_1.load_weights(f"{path}/critic_1.weights.h5")
        self.critic_2.load_weights(f"{path}/critic_2.weights.h5")
        self.target_critic_1.set_weights(self.critic_1.get_weights())
        self.target_critic_2.set_weights(self.critic_2.get_weights())

def train_agent():
    # Create and configure the environment
    env = gym.make('highway-v0', render_mode=None, config={
        'observation': {
            'type': 'Kinematics',
            'features': ['presence', 'x', 'y', 'vx', 'vy'],
            'absolute': True,
            'normalize': True,
        },
        'action': {
            'type': 'DiscreteMetaAction',
        },
        'lanes_count': 4,
        'vehicles_count': 50,
        'duration': 40,
        'initial_spacing': 2
    })
    
    # Print environment information for debugging
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Number of actions: {env.action_space.n}")
    
    # Get state and action dimensions
    state_dim = env.observation_space.shape  # Now passing full shape tuple
    action_dim = env.action_space.n
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Initialize the agent
    agent = DBSAC(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[256, 256],
        actor_lr=3e-4,
        critic_lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        buffer_capacity=100000,
        recent_size=10000,
        batch_size=256
    )
    
    # Training parameters
    max_episodes = 1000
    max_steps = 1000
    batch_size = 256
    eval_interval = 10
    
    # Training loop
    episode_rewards = []
    
    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Store transition in replay buffer
            agent.replay_buffer.add(state, action, reward, next_state, done)
            
            # Update the agent
            if len(agent.replay_buffer) > batch_size:
                metrics = agent.update(batch_size)
                
            episode_reward += reward
            state = next_state
            
            if done or truncated:
                break
                
        episode_rewards.append(episode_reward)
        
        # Print progress
        if (episode + 1) % eval_interval == 0:
            avg_reward = np.mean(episode_rewards[-eval_interval:])
            print(f"Episode {episode + 1}, Average Reward: {avg_reward:.2f}")
            
        # Save models periodically
        if (episode + 1) % 100 == 0:
            agent.save_models("./models")
            
    return agent, episode_rewards

def evaluate_agent(agent, num_episodes=10):
    env = gym.make('highway-v0', render_mode=None, config={
        'observation': {
            'type': 'Kinematics',
            'features': ['presence', 'x', 'y', 'vx', 'vy'],
            'absolute': True,
            'normalize': True,
        },
        'action': {
            'type': 'DiscreteMetaAction',
            'target_speeds': [20, 25, 30]  # Limit the speed actions
        },
        'lanes_count': 4,
        'vehicles_count': 50,
        'duration': 40,
        'initial_spacing': 2
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
    # Train the agent
    print("Starting training...")
    agent, training_rewards = train_agent()
    
    # Evaluate the trained agent
    print("\nStarting evaluation...")
    mean_eval_reward = evaluate_agent(agent)
    print(f"\nMean evaluation reward: {mean_eval_reward:.2f}") 
