# Import required libraries
import gymnasium as gym
import highway_env
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML, display
import os

# Mount Google Drive if needed (optional, since weights are in /content/sample_data)
try:
    from google.colab import drive
    drive.mount('/content/drive')
    print("Google Drive mounted successfully (optional)")
except ImportError:
    print("Not running in Colab environment")

# Define the DQN network architecture (must match your trained model)
class DQNNetwork(tf.keras.Model):
    def __init__(self, state_dim, action_dim, hidden_dims):
        super(DQNNetwork, self).__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.layers_list = []
        for dim in hidden_dims:
            self.layers_list.extend([
                tf.keras.layers.Dense(dim, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.1)
            ])
        self.q_head = tf.keras.layers.Dense(action_dim)
    
    def call(self, state):
        x = self.flatten(state)
        for layer in self.layers_list:
            x = layer(x)
        return self.q_head(x)

# Set up the highway environment with render_mode='rgb_array'
env = gym.make('highway-v0', render_mode='rgb_array', config={
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
    'collision_reward': -10.0,
    'high_speed_reward': 0.4,
    'right_lane_reward': 0.0,
    'lane_change_reward': 0.4,
    'reward_speed_range': [20, 30],
    'normalize_reward': True,
    'offroad_terminal': True,
    'simulation_frequency': 15,
    'policy_frequency': 5
})

# Get state and action dimensions
state_dim = env.observation_space.shape
action_dim = env.action_space.n

# Initialize and build the DQN model
model = DQNNetwork(state_dim, action_dim, hidden_dims=[256, 256])
dummy_state = tf.zeros((1,) + state_dim)
model(dummy_state)  # Build the model

# Load the trained model weights from your specified path
weights_path = "/content/sample_data/q_network.weights.h5"
model.load_weights(weights_path)
print(f"Loaded weights from {weights_path}")

# Function to select the best action based on Q-values
def select_action(state, model):
    state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
    q_values = model(state_tensor)[0]
    action = tf.argmax(q_values).numpy()
    return action

# Run the simulation for one episode and collect frames
state, _ = env.reset()
frames = []
episode_reward = 0
while True:
    action = select_action(state, model)
    next_state, reward, done, truncated, _ = env.step(action)
    frame = env.render()  # Capture the frame as an RGB array
    frames.append(frame)
    state = next_state
    episode_reward += reward
    if done or truncated:
        break
print(f"Episode Reward: {episode_reward:.2f}")

# Create an animation from the frames
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_axis_off()
im = ax.imshow(frames[0])

def update(frame):
    im.set_data(frame)
    return [im]

# Generate and display the animation
ani = animation.FuncAnimation(fig, update, frames=frames, interval=100, blit=True)
html5_video = ani.to_html5_video()
display(HTML(html5_video))

# Clean up by closing the environment
env.close()
