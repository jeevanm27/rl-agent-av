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
        # Base safety thresholds with more conservative values
        self.base_ttc_threshold = 3.0  # Increased from 2.0 for better safety margin
        self.base_headway_threshold = 2.0  # Increased from 1.5 for safer following distance
        self.base_safety_distance = 25.0  # Increased from 20.0 for better spacing
        
        # Lane change specific parameters
        self.min_lane_change_ttc = 4.0  # Minimum TTC required for lane change
        self.min_lane_change_space = 30.0  # Minimum space required for lane change
        self.lane_change_cooldown = 2.0  # Seconds to wait between lane changes
        
        # Adaptive weights with emphasis on speed and safety
        self.speed_weight = 0.35  # Increased emphasis on speed
        self.density_weight = 0.25  # Slightly reduced
        self.scenario_weight = 0.40  # Maintained for risk assessment
        
        # Safety reward weights with wider range
        self.min_safety_weight = 0.3  # Increased from 0.2
        self.max_safety_weight = 0.9  # Increased from 0.8
        
        # Historical tracking
        self.risk_history = deque(maxlen=100)
        self.avg_risk_level = 0.5
        self.last_lane_change_time = 0
        
    def compute_traffic_density(self, vehicles_info: List[Dict]) -> Tuple[float, Dict]:
        """Enhanced traffic density computation with per-lane analysis."""
        if not vehicles_info:
            return 0.0, {}
            
        # Initialize per-lane vehicle counts
        lane_counts = defaultdict(int)
        lane_speeds = defaultdict(list)
        
        # Count vehicles and average speeds per lane
        for vehicle in vehicles_info:
            lane = vehicle.get('lane_index', 0)
            if abs(vehicle.get('x', 0)) < 50:  # Within relevant range
                lane_counts[lane] += 1
                lane_speeds[lane].append(vehicle.get('vx', 0))
        
        # Compute average speed per lane
        lane_avg_speeds = {
            lane: np.mean(speeds) if speeds else 0
            for lane, speeds in lane_speeds.items()
        }
        
        # Overall density (normalized)
        total_density = sum(lane_counts.values()) / (10.0 * len(lane_counts))
        
        return min(total_density, 1.0), {
            'lane_counts': lane_counts,
            'lane_avg_speeds': lane_avg_speeds
        }
    
    def compute_scenario_risk(self, vehicles_info: List[Dict], ego_speed: float, ego_lane: int) -> Tuple[float, Dict]:
        """Enhanced scenario risk computation with lane-specific analysis."""
        if not vehicles_info:
            return 0.0, {}
            
        risks = {
            'front_risk': 0.0,
            'rear_risk': 0.0,
            'side_risks': {},
            'lane_change_risks': {}
        }
        
        for vehicle in vehicles_info:
            rel_x = vehicle.get('x', 0)
            rel_y = vehicle.get('y', 0)
            rel_speed = ego_speed - vehicle.get('vx', 0)
            vehicle_lane = vehicle.get('lane_index', 0)
            
            # Compute distance and time-to-collision
            distance = abs(rel_x)
            ttc = distance / abs(rel_speed) if abs(rel_speed) > 0.1 else float('inf')
            
            # Categorize risk based on relative position
            if abs(rel_y) < 1.0:  # Same lane
                if rel_x > 0:  # Vehicle ahead
                    risks['front_risk'] = max(risks['front_risk'], 
                                            self._compute_individual_risk(distance, rel_speed, ttc))
                else:  # Vehicle behind
                    risks['rear_risk'] = max(risks['rear_risk'], 
                                           self._compute_individual_risk(distance, rel_speed, ttc))
            else:  # Adjacent lanes
                lane_diff = int(round(rel_y))
                risks['side_risks'][ego_lane + lane_diff] = max(
                    risks['side_risks'].get(ego_lane + lane_diff, 0.0),
                    self._compute_individual_risk(distance, rel_speed, ttc) * 0.7  # Reduced weight for side vehicles
                )
        
        # Compute lane change risks
        for target_lane in [ego_lane - 1, ego_lane + 1]:
            if target_lane in risks['side_risks']:
                risks['lane_change_risks'][target_lane] = max(
                    risks['side_risks'][target_lane],
                    risks['front_risk'] * 0.5,
                    risks['rear_risk'] * 0.5
                )
        
        # Overall risk is maximum of all risks
        max_risk = max(
            [risks['front_risk'], risks['rear_risk']] +
            list(risks['side_risks'].values())
        )
        
        return min(max_risk, 1.0), risks
    
    def _compute_individual_risk(self, distance: float, rel_speed: float, ttc: float) -> float:
        """Helper function to compute risk for individual vehicle interactions."""
        # Base risk from distance
        distance_risk = 1.0 / (1.0 + distance/20.0)
        
        # Speed risk increases with closing speed
        speed_risk = abs(rel_speed) / 30.0 if rel_speed < 0 else 0.0
        
        # TTC risk
        ttc_risk = 1.0 / (1.0 + ttc) if ttc != float('inf') else 0.0
        
        # Combine risks with weights
        return min(0.4 * distance_risk + 0.3 * speed_risk + 0.3 * ttc_risk, 1.0)
    
    def get_adaptive_thresholds(self, ego_speed: float, vehicles_info: List[Dict], ego_lane: int, time_elapsed: float) -> Dict[str, float]:
        """Enhanced adaptive thresholds with lane change considerations."""
        # Normalize ego speed (assuming max speed of 30 m/s)
        norm_speed = min(ego_speed / 30.0, 1.0)
        
        # Compute enhanced traffic density and scenario risk
        traffic_density, lane_info = self.compute_traffic_density(vehicles_info)
        scenario_risk, risk_info = self.compute_scenario_risk(vehicles_info, ego_speed, ego_lane)
        
        # Compute overall risk factor
        risk_factor = (
            self.speed_weight * norm_speed +
            self.density_weight * traffic_density +
            self.scenario_weight * scenario_risk
        )
        
        # Update risk history and average
        self.risk_history.append(risk_factor)
        self.avg_risk_level = np.mean(list(self.risk_history))
        
        # Compute lane change availability
        lane_change_available = (time_elapsed - self.last_lane_change_time) > self.lane_change_cooldown
        
        # Adjust thresholds based on risk factor
        ttc_threshold = self.base_ttc_threshold * (1.0 + risk_factor)
        headway_threshold = self.base_headway_threshold * (1.0 + risk_factor)
        safety_distance = self.base_safety_distance * (1.0 + risk_factor)
        
        # Compute adaptive safety weight
        safety_weight = self.min_safety_weight + (
            self.max_safety_weight - self.min_safety_weight
        ) * self.avg_risk_level
        
        return {
            'ttc_threshold': ttc_threshold,
            'headway_threshold': headway_threshold,
            'safety_distance': safety_distance,
            'safety_weight': safety_weight,
            'risk_factor': risk_factor,
            'lane_info': lane_info,
            'risk_info': risk_info,
            'lane_change_available': lane_change_available
        }
    
    def compute_safety_reward(self, 
                            ego_speed: float,
                            vehicles_info: List[Dict],
                            ttc: float,
                            headway: float,
                            ego_lane: int,
                            action: int,
                            time_elapsed: float) -> Tuple[float, Dict]:
        """Enhanced safety reward computation with lane change incentives."""
        thresholds = self.get_adaptive_thresholds(ego_speed, vehicles_info, ego_lane, time_elapsed)
        
        # Basic safety violations
        ttc_violation = max(0, 1 - ttc/thresholds['ttc_threshold'])
        headway_violation = max(0, 1 - headway/thresholds['headway_threshold'])
        
        # Lane change analysis
        lane_info = thresholds['lane_info']
        risk_info = thresholds['risk_info']
        
        # Initialize reward components
        safety_penalty = 0.0
        speed_reward = 0.0
        lane_change_reward = 0.0
        
        # Safety penalty based on violations
        safety_penalty = -(
            ttc_violation * thresholds['safety_weight'] +
            headway_violation * thresholds['safety_weight']
        )
        
        # Speed reward
        target_speed = 30.0  # m/s
        speed_reward = 0.4 * (ego_speed / target_speed) * (1 - thresholds['risk_factor'])
        
        # Lane change reward
        if action in [0, 2]:  # Lane change actions
            if thresholds['lane_change_available']:
                target_lane = ego_lane + (1 if action == 2 else -1)
                
                # Check if lane change is beneficial
                current_lane_density = lane_info['lane_counts'].get(ego_lane, 0)
                target_lane_density = lane_info['lane_counts'].get(target_lane, 0)
                
                if target_lane_density < current_lane_density:
                    # Reward lane change if target lane is less congested
                    lane_change_reward = 0.3 * (1 - risk_info['lane_change_risks'].get(target_lane, 1.0))
                    
                    # Update last lane change time if action is executed
                    self.last_lane_change_time = time_elapsed
        
        # Combine rewards
        total_reward = safety_penalty + speed_reward + lane_change_reward
        
        return total_reward, {
            **thresholds,
            'safety_penalty': safety_penalty,
            'speed_reward': speed_reward,
            'lane_change_reward': lane_change_reward
        }

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
        batch_size: int = 256,
        checkpoint_dir: str = "./checkpoints"
    ):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.checkpoint_dir = checkpoint_dir
        
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
            
    def save_models(self, path: str, save_buffer: bool = True, episode: int = None, metrics: Dict = None):
        """Save model weights, replay buffer, and training metrics."""
        if not os.path.exists(path):
            os.makedirs(path)
            
        # Save model weights
        self.actor.save_weights(f"{path}/actor.weights.h5")
        self.critic_1.save_weights(f"{path}/critic_1.weights.h5")
        self.critic_2.save_weights(f"{path}/critic_2.weights.h5")
        
        # Save replay buffer if requested
        if save_buffer:
            buffer_path = os.path.join(path, "replay_buffer.pkl")
            with open(buffer_path, 'wb') as f:
                pickle.dump(self.replay_buffer, f)
        
        # Save training metrics and episode number if provided
        if episode is not None or metrics is not None:
            metadata = {
                'episode': episode,
                'metrics': metrics or {}
            }
            with open(os.path.join(path, "training_metadata.json"), 'w') as f:
                json.dump(metadata, f)
        
    def load_models(self, path: str, load_buffer: bool = True):
        """Load model weights and optionally replay buffer."""
        # Load model weights
        self.actor.load_weights(f"{path}/actor.weights.h5")
        self.critic_1.load_weights(f"{path}/critic_1.weights.h5")
        self.critic_2.load_weights(f"{path}/critic_2.weights.h5")
        self.target_critic_1.set_weights(self.critic_1.get_weights())
        self.target_critic_2.set_weights(self.critic_2.get_weights())
        
        # Load replay buffer if requested and exists
        if load_buffer and os.path.exists(os.path.join(path, "replay_buffer.pkl")):
            with open(os.path.join(path, "replay_buffer.pkl"), 'rb') as f:
                self.replay_buffer = pickle.load(f)
                
        # Load and return training metadata if exists
        metadata_path = os.path.join(path, "training_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return None

def train_agent(checkpoint_dir="/content/drive/MyDrive/highway_rl_checkpoints", start_episode=0, max_episodes=1000):
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
            'target_speeds': [25, 30],  # Encourage higher speeds
            'longitudinal': True,
            'lateral': True,
        },
        'lanes_count': 4,
        'vehicles_count': 50,
        'duration': 40,
        'initial_spacing': 2,
        'collision_reward': -2.0,  # Increased collision penalty
        'high_speed_reward': 0.4,
        'right_lane_reward': 0.1,
        'lane_change_reward': 0.2,
        'reward_speed_range': [25, 30],  # Increased target speed range
        'normalize_reward': True,
        'offroad_terminal': True,
        'simulation_frequency': 15,
        'policy_frequency': 5,
        'screen_width': 800,
        'screen_height': 600,
        'centering_position': [0.3, 0.5],
        'scaling': 5.5,
        'show_trajectories': True
    })
    
    # Get state and action dimensions
    state_dim = env.observation_space.shape
    action_dim = env.action_space.n
    
    # Initialize agent
    agent = DBSAC(
        state_dim=state_dim,
        action_dim=action_dim,
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
                
                # Load best reward if available
                best_avg_reward = metadata.get('metrics', {}).get('best_avg_reward', float('-inf'))
                print(f"Previous best average reward: {best_avg_reward:.2f}")
    
    # Training parameters
    max_steps = 1000
    batch_size = 256
    eval_interval = 10
    checkpoint_interval = 100  # Save checkpoint every 100 episodes
    
    # Training metrics
    episode_rewards = []
    best_avg_reward = float('-inf')
    training_metrics = {'episode_rewards': [], 'avg_rewards': []}
    
    print(f"Starting training from episode {start_episode}")
    print(f"Checkpoints will be saved every {checkpoint_interval} episodes to: {checkpoint_dir}")
    
    try:
        for episode in range(start_episode, max_episodes):
            state, _ = env.reset()
            episode_reward = 0
            episode_start_time = time.time()
            
            for step in range(max_steps):
                action = agent.select_action(state)
                next_state, reward, done, truncated, info = env.step(action)
                
                # Extract enhanced state information
                ego_speed = info.get('speed', 0)
                vehicles_info = info.get('vehicles_info', [])
                ttc = info.get('time_to_collision', float('inf'))
                headway = info.get('headway', float('inf'))
                ego_lane = info.get('lane_index', 0)
                time_elapsed = time.time() - episode_start_time
                
                # Compute enhanced safety reward
                safety_reward, metrics = safety_module.compute_safety_reward(
                    ego_speed, vehicles_info, ttc, headway, ego_lane, action, time_elapsed
                )
                
                # Combine rewards with emphasis on speed and safe lane changes
                combined_reward = (
                    0.4 * reward +  # Original environment reward
                    0.4 * safety_reward +  # Safety considerations
                    0.2 * metrics['speed_reward']  # Additional speed incentive
                )
                
                # Store transition with enhanced reward
                agent.replay_buffer.add(state, action, combined_reward, next_state, done)
                
                if len(agent.replay_buffer) > batch_size:
                    metrics = agent.update(batch_size)
                    
                episode_reward += combined_reward
                state = next_state
                
                if done or truncated:
                    break
                    
            episode_rewards.append(episode_reward)
            training_metrics['episode_rewards'].append(episode_reward)
            
            # Print progress
            if (episode + 1) % eval_interval == 0:
                avg_reward = np.mean(episode_rewards[-eval_interval:])
                training_metrics['avg_rewards'].append(avg_reward)
                print(f"Episode {episode + 1}, Average Reward: {avg_reward:.2f}")
                
                # Update best model if we have a new best performance
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    best_model_path = os.path.join(checkpoint_dir, "best_model")
                    
                    # Save best model with temporary name first
                    temp_best_path = os.path.join(checkpoint_dir, "temp_best_model")
                    agent.save_models(
                        temp_best_path,
                        save_buffer=True,
                        episode=episode,
                        metrics={'avg_reward': avg_reward, 'best_avg_reward': best_avg_reward}
                    )
                    
                    # Safely replace old best model
                    if os.path.exists(best_model_path):
                        shutil.rmtree(best_model_path)
                    os.rename(temp_best_path, best_model_path)
                    print(f"New best model saved with average reward: {avg_reward:.2f}")
            
            # Save checkpoint every 100 episodes
            if (episode + 1) % checkpoint_interval == 0:
                # Save with temporary name first
                temp_checkpoint_path = os.path.join(checkpoint_dir, f"temp_checkpoint_{episode + 1}")
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{episode + 1}")
                
                agent.save_models(
                    temp_checkpoint_path,
                    save_buffer=True,
                    episode=episode,
                    metrics={
                        'episode_rewards': episode_rewards,
                        'best_avg_reward': best_avg_reward,
                        'training_metrics': training_metrics
                    }
                )
                
                # Safely replace old checkpoint
                if os.path.exists(checkpoint_path):
                    shutil.rmtree(checkpoint_path)
                os.rename(temp_checkpoint_path, checkpoint_path)
                
                # Keep only the latest checkpoint and best model
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
            metrics={
                'episode_rewards': episode_rewards,
                'best_avg_reward': best_avg_reward,
                'training_metrics': training_metrics
            }
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
                    'best_avg_reward': best_avg_reward,
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
    # Set up checkpoint directory in Google Drive
    CHECKPOINT_DIR = "/content/drive/MyDrive/highway_rl_checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    print("Starting training...")
    print(f"Checkpoints will be saved to: {CHECKPOINT_DIR}")
    
    agent, training_rewards = train_agent(checkpoint_dir=CHECKPOINT_DIR)
    
    # Evaluate the trained agent
    print("\nStarting evaluation...")
    mean_eval_reward = evaluate_agent(agent)
    print(f"\nMean evaluation reward: {mean_eval_reward:.2f}") 
