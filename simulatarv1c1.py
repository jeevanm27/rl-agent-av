import tensorflow as tf
import numpy as np
import gymnasium as gym
import highway_env
from typing import Dict, List, Tuple
import time

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
                    kernel_initializer='glorot_uniform'
                ),
                tf.keras.layers.LayerNormalization()  # Changed from BatchNorm to LayerNorm
            ])
            
        # Final layer for Q-values
        self.q_head = tf.keras.layers.Dense(
            action_dim,
            kernel_initializer='glorot_uniform'
        )
        
    def call(self, state):
        x = self.flatten(state)
        for layer in self.layers_list:
            x = layer(x)
        return self.q_head(x)

def create_env(render_mode='rgb_array'):
    """Create the highway environment with specific settings."""
    env = gym.make('highway-v0', render_mode=render_mode, config={
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
        'policy_frequency': 5,
        'screen_width': 600,
        'screen_height': 400,
        'centering_position': [0.3, 0.5],
        'scaling': 5.5,
        'show_trajectories': True
    })
    return env

def evaluate_model(model_path: str, num_episodes: int = 5, render: bool = True, save_video: bool = False):
    """
    Evaluate a trained model on the highway environment.
    
    Args:
        model_path: Path to the trained model weights
        num_episodes: Number of episodes to evaluate
        render: Whether to render the environment
        save_video: Whether to save the evaluation as a video
    """
    # Create environment
    env = create_env(render_mode='rgb_array' if render or save_video else None)
    
    # Get state and action dimensions
    state_dim = env.observation_space.shape
    action_dim = env.action_space.n
    
    # Create and load model
    model = DQNNetwork(state_dim, action_dim, hidden_dims=[256, 256])
    
    # Build model by calling it once
    dummy_state = tf.zeros((1,) + state_dim)
    model(dummy_state)
    
    # Load weights
    try:
        # Try to load weights directly
        model.load_weights(model_path)
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        try:
            # If direct loading fails, try loading as a saved model
            model = tf.keras.models.load_model(model_path)
            print(f"Successfully loaded saved model from {model_path}")
        except Exception as e:
            print(f"Error loading saved model: {str(e)}")
            return
    
    # Evaluation metrics
    episode_rewards = []
    episode_lengths = []
    collision_count = 0
    success_count = 0
    
    if save_video:
        from gymnasium.wrappers import RecordVideo
        video_path = "videos/highway-evaluation"
        print(f"Videos will be saved to {video_path}")
        env = RecordVideo(env, video_path)
    
    try:
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            steps = 0
            done = False
            
            while not done:
                # Get action from model
                state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
                q_values = model(state_tensor)
                action = int(tf.argmax(q_values[0]))
                
                # Take step in environment
                next_state, reward, done, truncated, info = env.step(action)
                
                if render and not save_video:
                    env.render()
                    time.sleep(0.1)  # Add delay to make visualization viewable
                
                episode_reward += reward
                state = next_state
                steps += 1
                
                # Check for collision or success
                if info.get('crashed', False):
                    collision_count += 1
                if steps >= env.config['duration'] * env.config['policy_frequency']:
                    success_count += 1
                
                if done or truncated:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)
            
            print(f"Episode {episode + 1}/{num_episodes}, "
                  f"Reward: {episode_reward:.2f}, "
                  f"Steps: {steps}, "
                  f"{'Collision' if info.get('crashed', False) else 'Success'}")
    
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
    
    finally:
        env.close()
        
        if episode_rewards:  # Only print summary if we have data
            print("\nEvaluation Summary:")
            print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
            print(f"Average Episode Length: {np.mean(episode_lengths):.1f} steps")
            print(f"Success Rate: {success_count/len(episode_rewards)*100:.1f}%")
            print(f"Collision Rate: {collision_count/len(episode_rewards)*100:.1f}%")
            
            if save_video:
                print(f"\nVideos saved in '{video_path}' directory")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate a trained highway environment model')
    parser.add_argument('--model_path', type=str, default="/content/checkpoints/q_network.weights (2).h5",
                        help='Path to the trained model weights')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes to evaluate')
    parser.add_argument('--render', action='store_true',
                        help='Whether to render the environment')
    parser.add_argument('--save_video', action='store_true',
                        help='Whether to save evaluation videos')
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model_path,
        num_episodes=args.episodes,
        render=args.render,
        save_video=args.save_video
    ) 
