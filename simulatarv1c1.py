# Install required packages
!pip install gymnasium
!pip install highway-env
!apt-get update
!apt-get install -y python-opengl
!apt-get install -y ffmpeg
!apt-get install -y xvfb
!pip install pyvirtualdisplay
!pip install imageio

# Import required libraries
import gymnasium as gym
import highway_env
import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple
import time
from pyvirtualdisplay import Display
from IPython.display import HTML
from base64 import b64encode
import os
import imageio

# Start virtual display
virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()

# Define model architecture
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

class DBSAC:
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256]):
        self.actor = Actor(state_dim, action_dim, hidden_dims)
        
        # Build model by calling it once
        dummy_state = tf.zeros((1,) + state_dim)
        self.actor(dummy_state)
    
    def load_models(self, path: str):
        self.actor.load_weights(f"{path}/actor.weights.h5")
    
    def select_action(self, state, evaluate=True):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        action_probs = self.actor.get_action_probs(state)[0]
        
        if evaluate:
            action = tf.argmax(action_probs).numpy()
        else:
            action_probs = tf.reshape(action_probs, (1, -1))
            action = tf.random.categorical(tf.math.log(action_probs + 1e-8), 1)[0, 0].numpy()
        
        return int(action)

def record_video(env, agent, video_length=500):
    """Record agent's behavior and return video as HTML."""
    frames = []
    state, _ = env.reset()
    done = False
    
    for _ in range(video_length):
        frames.append(env.render())
        
        action = agent.select_action(state, evaluate=True)
        state, reward, done, truncated, _ = env.step(action)
        
        if done or truncated:
            break
    
    env.close()
    
    # Convert frames to video
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        writer = imageio.get_writer(f.name, fps=15)
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        
        # Read the video file and encode it to base64
        with open(f.name, 'rb') as video_file:
            video_data = video_file.read()
    
    # Clean up temporary file
    os.unlink(f.name)
    
    # Encode video in base64 and create HTML
    video = b64encode(video_data).decode()
    return HTML(f'<video width="800" height="600" controls><source src="data:video/mp4;base64,{video}" type="video/mp4"></video>')

def evaluate_episodes(env, agent, num_episodes=5):
    total_rewards = []
    total_speeds = []
    total_lane_changes = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_speeds = []
        lane_changes = 0
        last_lane = None
        done = False
        
        while not done:
            action = agent.select_action(state, evaluate=True)
            next_state, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            if 'speed' in info:
                episode_speeds.append(info['speed'])
            
            current_lane = info.get('lane_index', None)
            if last_lane is not None and current_lane != last_lane:
                lane_changes += 1
            last_lane = current_lane
            
            state = next_state
            if truncated:
                break
        
        total_rewards.append(episode_reward)
        if episode_speeds:
            total_speeds.append(np.mean(episode_speeds))
        total_lane_changes.append(lane_changes)
        
        print(f"Episode {episode + 1} Statistics:")
        print(f"- Total Reward: {episode_reward:.2f}")
        if episode_speeds:
            print(f"- Average Speed: {np.mean(episode_speeds):.2f} m/s")
        print(f"- Lane Changes: {lane_changes}\n")
    
    print("\nOverall Statistics:")
    print(f"Average Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    if total_speeds:
        print(f"Average Speed: {np.mean(total_speeds):.2f} ± {np.std(total_speeds):.2f} m/s")
    print(f"Average Lane Changes: {np.mean(total_lane_changes):.2f} ± {np.std(total_lane_changes):.2f}")

# Create a directory for models if it doesn't exist
!mkdir -p /content/models

print("\nPlease upload your model files to /content/models/")
print("Required files: actor.weights.h5")

# Create and configure environment
env = gym.make('highway-v0', render_mode='rgb_array', config={
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
    'initial_spacing': 2,
    'collision_reward': -1.0,
    'high_speed_reward': 0.4,
    'right_lane_reward': 0.1,
    'lane_change_reward': 0.2,
    'reward_speed_range': [20, 30],
    'normalize_reward': True,
    'offroad_terminal': True,
    'screen_width': 800,
    'screen_height': 600,
    'centering_position': [0.3, 0.5],
    'scaling': 5.5,
    'show_trajectories': True
})

# Initialize agent
state_dim = env.observation_space.shape
action_dim = env.action_space.n
agent = DBSAC(state_dim, action_dim)

# Try to load model from different possible locations
model_paths = [
    '/content/models/models_best',
    '/content/models',
    'models_best',
    'models'
]

model_loaded = False
for path in model_paths:
    try:
        if os.path.exists(os.path.join(path, 'actor.weights.h5')):
            agent.load_models(path)
            print(f"Successfully loaded model from {path}")
            model_loaded = True
            break
    except Exception as e:
        continue

if not model_loaded:
    print("\nError: Could not find model files!")
    print("Please make sure you have uploaded actor.weights.h5 to one of these locations:")
    for path in model_paths:
        print(f"- {path}")
else:
    # Generate and display video
    print("\nGenerating video of agent's performance...")
    video = record_video(env, agent)
    display(video)
    
    # Run evaluation
    print("\nRunning detailed evaluation...")
    evaluate_episodes(env, agent)
