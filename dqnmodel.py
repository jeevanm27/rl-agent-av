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
        # Initialize default return values
        default_lane_info = {
            'lane_counts': defaultdict(int),
            'lane_avg_speeds': defaultdict(float)
        }
        
        if not vehicles_info:
            return 0.0, default_lane_info
            
        # Initialize per-lane vehicle counts
        lane_counts = defaultdict(int)
        lane_speeds = defaultdict(list)
        
        try:
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
            total_density = sum(lane_counts.values()) / (10.0 * max(len(lane_counts), 1))
            
            return min(total_density, 1.0), {
                'lane_counts': lane_counts,
                'lane_avg_speeds': lane_avg_speeds
            }
        except Exception as e:
            print(f"Warning: Error in compute_traffic_density: {str(e)}")
            return 0.0, default_lane_info
    
    def compute_scenario_risk(self, vehicles_info: List[Dict], ego_speed: float, ego_lane: int) -> Tuple[float, Dict]:
        """Enhanced scenario risk computation with lane-specific analysis."""
        # Initialize default return values
        default_risk_info = {
            'front_risk': 0.0,
            'rear_risk': 0.0,
            'side_risks': {},
            'lane_change_risks': {}
        }
        
        if not vehicles_info:
            return 0.0, default_risk_info
            
        risks = default_risk_info.copy()
        
        try:
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
                        self._compute_individual_risk(distance, rel_speed, ttc) * 0.7
                    )
            
            # Compute lane change risks
            for target_lane in [ego_lane - 1, ego_lane + 1]:
                if target_lane in risks['side_risks']:
                    risks['lane_change_risks'][target_lane] = max(
                        risks['side_risks'][target_lane],
                        risks['front_risk'] * 0.5,
                        risks['rear_risk'] * 0.5
                    )
            
            max_risk = max(
                [risks['front_risk'], risks['rear_risk']] +
                list(risks['side_risks'].values() or [0.0])
            )
            
            return min(max_risk, 1.0), risks
            
        except Exception as e:
            print(f"Warning: Error in compute_scenario_risk: {str(e)}")
            return 0.0, default_risk_info
    
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
        try:
            thresholds = self.get_adaptive_thresholds(ego_speed, vehicles_info, ego_lane, time_elapsed)
            
            # Basic safety violations
            ttc_violation = max(0, 1 - ttc/thresholds['ttc_threshold'])
            headway_violation = max(0, 1 - headway/thresholds['headway_threshold'])
            
            # Lane change analysis
            lane_info = thresholds.get('lane_info', {'lane_counts': defaultdict(int)})
            risk_info = thresholds.get('risk_info', {'lane_change_risks': {}})
            
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
                if thresholds.get('lane_change_available', False):
                    target_lane = ego_lane + (1 if action == 2 else -1)
                    
                    # Check if lane change is beneficial
                    current_lane_density = lane_info['lane_counts'].get(ego_lane, 0)
                    target_lane_density = lane_info['lane_counts'].get(target_lane, 0)
                    
                    if target_lane_density < current_lane_density:
                        # Reward lane change if target lane is less congested
                        lane_change_reward = 0.3 * (1 - risk_info.get('lane_change_risks', {}).get(target_lane, 1.0))
                        
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
            
        except Exception as e:
            print(f"Warning: Error in compute_safety_reward: {str(e)}")
            # Return safe default values
            return 0.0, {
                'ttc_threshold': self.base_ttc_threshold,
                'headway_threshold': self.base_headway_threshold,
                'safety_distance': self.base_safety_distance,
                'safety_weight': self.min_safety_weight,
                'risk_factor': 0.5,
                'safety_penalty': 0.0,
                'speed_reward': 0.0,
                'lane_change_reward': 0.0
            }

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)

class DQNNetwork(tf.keras.Model):
    def __init__(self, state_dim: Tuple[int, ...], action_dim: int, hidden_dims: List[int]):
        super(DQNNetwork, self).__init__()
        
        # Flatten layer to handle 2D state input
        self.flatten = tf.keras.layers.Flatten()
        
        self.layers_list = []
        for dim in hidden_dims:
            self.layers_list.extend([
                tf.keras.layers.Dense(dim, activation='relu'),
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.Dropout(0.1)  # Add dropout for regularization
            ])
            
        # Final layer for Q-values
        self.q_head = tf.keras.layers.Dense(action_dim)
        
    def call(self, state):
        x = self.flatten(state)
        for layer in self.layers_list:
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
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        tau: float = 0.005,
        buffer_capacity: int = 100000,
        batch_size: int = 64,
        update_target_every: int = 100,
        checkpoint_dir: str = "./checkpoints"
    ):
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.tau = tau
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.update_target_every = update_target_every
        self.checkpoint_dir = checkpoint_dir
        self.train_step_counter = 0
        
        # Initialize networks
        self.q_network = DQNNetwork(state_dim, action_dim, hidden_dims)
        self.target_network = DQNNetwork(state_dim, action_dim, hidden_dims)
        
        # Build models by calling them once
        dummy_state = tf.zeros((1,) + state_dim)
        self.q_network(dummy_state)
        self.target_network(dummy_state)
        
        # Copy weights to target network
        self.target_network.set_weights(self.q_network.get_weights())
        
        # Initialize optimizer with gradient clipping
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # Initialize Huber loss for more stable training
        self.loss_fn = tf.keras.losses.Huber()
        
    def _update_target(self):
        """Soft update target network weights."""
        for target_weights, q_weights in zip(self.target_network.weights, self.q_network.weights):
            target_weights.assign(self.tau * q_weights + (1 - self.tau) * target_weights)
            
    def update(self, batch_size: int) -> Dict[str, float]:
        if len(self.replay_buffer) < batch_size:
            return {}
            
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Convert to tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        
        with tf.GradientTape() as tape:
            # Compute Q values for current states
            q_values = self.q_network(states)
            q_values_selected = tf.gather(q_values, actions, batch_dims=1)
            
            # Compute target Q values
            next_q_values = self.target_network(next_states)
            next_q_values_max = tf.reduce_max(next_q_values, axis=1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values_max
            
            # Compute loss using Huber loss
            loss = self.loss_fn(target_q_values, q_values_selected)
        
        # Compute and apply gradients with clipping
        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        gradients = [tf.clip_by_norm(g, 1.0) for g in gradients]
        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))
        
        # Periodically update target network
        self.train_step_counter += 1
        if self.train_step_counter % self.update_target_every == 0:
            self._update_target()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return {
            'loss': float(loss),
            'epsilon': float(self.epsilon)
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
            'target_speeds': [25, 30],
            'longitudinal': True,
            'lateral': True,
        },
        'lanes_count': 4,
        'vehicles_count': 50,
        'duration': 40,
        'initial_spacing': 2,
        'collision_reward': -2.0,
        'high_speed_reward': 0.4,
        'right_lane_reward': 0.1,
        'lane_change_reward': 0.2,
        'reward_speed_range': [25, 30],
        'normalize_reward': True,
        'offroad_terminal': True,
        'simulation_frequency': 15,
        'policy_frequency': 5
    })
    
    # Get state and action dimensions
    state_dim = env.observation_space.shape
    action_dim = env.action_space.n
    
    # Initialize agent with optimized hyperparameters for DQN
    agent = DQN(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[256, 256],  # Keeping the same architecture
        learning_rate=2.5e-4,    # Slightly higher learning rate for DQN
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        tau=0.005,
        buffer_capacity=50000,   # Smaller buffer size for faster learning
        batch_size=64,           # Smaller batch size for faster updates
        update_target_every=100, # Update target network every 100 steps
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
    checkpoint_interval = 50  # Save checkpoints more frequently
    min_samples = 1000  # Minimum samples before starting training
    
    # Training metrics
    episode_rewards = []
    training_metrics = {
        'episode_rewards': [],
        'avg_rewards': [],
        'losses': [],
        'epsilons': []
    }
    
    print(f"Starting training from episode {start_episode}")
    print(f"Checkpoints will be saved every {checkpoint_interval} episodes to: {checkpoint_dir}")
    
    try:
        for episode in range(start_episode, max_episodes):
            state, _ = env.reset()
            episode_reward = 0
            episode_start_time = time.time()
            episode_losses = []
            
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
                
                # Store transition
                agent.replay_buffer.add(state, action, combined_reward, next_state, done)
                
                # Update network if we have enough samples
                if len(agent.replay_buffer) > min_samples:
                    update_metrics = agent.update(batch_size)
                    if update_metrics:
                        episode_losses.append(update_metrics['loss'])
                
                episode_reward += combined_reward
                state = next_state
                
                if done or truncated:
                    break
            
            # Store episode metrics
            episode_rewards.append(episode_reward)
            training_metrics['episode_rewards'].append(episode_reward)
            if episode_losses:
                training_metrics['losses'].append(np.mean(episode_losses))
            training_metrics['epsilons'].append(agent.epsilon)
            
            # Print progress
            if (episode + 1) % eval_interval == 0:
                avg_reward = np.mean(episode_rewards[-eval_interval:])
                training_metrics['avg_rewards'].append(avg_reward)
                print(f"Episode {episode + 1}, "
                      f"Average Reward: {avg_reward:.2f}, "
                      f"Epsilon: {agent.epsilon:.3f}, "
                      f"Loss: {np.mean(episode_losses):.3f}")
            
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
        },
        'action': {
            'type': 'DiscreteMetaAction',
            'target_speeds': [20, 25, 30]  
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
