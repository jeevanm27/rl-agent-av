# Import required libraries
import gymnasium as gym
import highway_env
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML, display  # Fix: Import display explicitly

# Define the DQN network architecture (must match the trained model)
class DQNNetwork(tf.keras.Model):
    def __init__(self, state_dim, action_dim, hidden_dims):
        super(DQNNetwork, self).__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.layers_list = []
        for dim in hidden_dims:
            self.layers_list.extend([
                tf.keras.layers.Dense(dim, activation='relu'),
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.Dropout(0.1)
            ])
        self.q_head = tf.keras.layers.Dense(action_dim)
    
    def call(self, state):
        x = self.flatten(state)
        for layer in self.layers_list:
            x = layer(x)
        return self.q_head(x)

# Set up the highway environment with render_mode='rgb_array'
try:
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
except Exception as e:
    print(f"Error initializing environment: {e}")
    raise

# Get state and action dimensions
state_dim = env.observation_space.shape
action_dim = env.action_space.n

# Initialize and build the DQN model
try:
    model = DQNNetwork(state_dim, action_dim, hidden_dims=[256, 256])
    dummy_state = tf.zeros((1,) + state_dim)
    model(dummy_state)
except Exception as e:
    print(f"Error initializing model: {e}")
    raise

# Load the trained model weights
try:
    model.load_weights("/content/sample_data/q_network.weights.h5")
except Exception as e:
    print(f"Error loading model weights: {e}")
    raise

# Function to select the best action based on Q-values
def select_action(state, model):
    try:
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        q_values = model(state_tensor)[0]
        action = tf.argmax(q_values).numpy()
        return action
    except Exception as e:
        print(f"Error selecting action: {e}")
        return 0  # Default action (e.g., idle)

# Run the simulation for one episode and collect frames
try:
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
except Exception as e:
    print(f"Error during simulation: {e}")
    env.close()
    raise

# Create an animation from the frames
try:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_axis_off()
    im = ax.imshow(frames[0])

    def update(frame):
        im.set_data(frame)
        return [im]

    # Generate the animation with a 100ms interval
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=100, blit=True)

    # Display the animation in Colab
    html5_video = ani.to_html5_video()
    display(HTML(html5_video))
except Exception as e:
    print(f"Error creating or displaying animation: {e}")
finally:
    # Clean up by closing the environment
    env.close()
