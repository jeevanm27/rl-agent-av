!pip install gymnasium highway-env pyvirtualdisplay > /dev/null 2>&1
!apt-get install -y xvfb python-opengl ffmpeg > /dev/null 2>&1

import gymnasium as gym
import highway_env
from pyvirtualdisplay import Display
from IPython import display as ipythondisplay
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time
import os

# ================== SETUP ==================
# Initialize virtual display
display = Display(visible=0, size=(1000, 800))
display.start()

# ================== LONG-DURATION ENVIRONMENT ==================
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
    'vehicles_count': 20,
    'duration': 200,  # Increased from 40 to 200 timesteps
    'initial_spacing': 2,
    'policy_frequency': 5,
    'simulation_frequency': 15,
    'offscreen_rendering': False,
    'show_trajectories': False
})

# ================== MODEL LOADING ==================
class HighwayAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self._build_network()

    def _build_network(self):
        """Build network matching your exact training architecture"""
        inputs = tf.keras.Input(shape=self.state_dim)
        x = tf.keras.layers.Flatten()(inputs)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.LayerNormalization()(x)
        outputs = tf.keras.layers.Dense(self.action_dim)(x)

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # Build model by running a dummy input
        dummy_input = tf.ones((1,) + self.state_dim)
        _ = self.model(dummy_input)

    def load_saved_weights(self, path):
        """Special weight loading that preserves layer names"""
        try:
            # Create a temporary model with the same architecture
            temp_model = tf.keras.models.clone_model(self.model)
            temp_model.build((None,) + self.state_dim)

            # Load weights into temporary model
            temp_model.load_weights(f"{path}/actor.weights.h5")

            # Transfer weights to main model
            for main_layer, temp_layer in zip(self.model.layers, temp_model.layers):
                main_layer.set_weights(temp_layer.get_weights())

            print("✅ Successfully loaded your trained model weights!")
            return True
        except Exception as e:
            print(f"❌ Error loading weights: {str(e)}")
            return False

    def select_action(self, state):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        logits = self.model(state)
        return tf.argmax(logits[0]).numpy()

# ================== INITIALIZATION ==================
state_dim = env.observation_space.shape
action_dim = env.action_space.n
print(f"State dimensions: {state_dim}, Action dimensions: {action_dim}")

agent = HighwayAgent(state_dim, action_dim)

# Load your trained weights
model_path = "./models"
print("\nLoading your trained model weights from:", os.path.abspath(model_path))
print("Contents of models directory:")
!ls -l {model_path}

if not agent.load_saved_weights(model_path):
    print("\n⚠️ Using random policy as fallback")
    print("Your trained weights couldn't be loaded, but you'll still see visualization")

# ================== LONG-RUN VISUALIZATION ==================
def run_long_simulation(agent, env, duration_seconds=60):
    """Run simulation for specified duration with efficient rendering"""
    state, _ = env.reset()
    start_time = time.time()
    frame_count = 0

    # Setup visualization
    plt.figure(figsize=(10, 6))
    img = plt.imshow(env.render())
    plt.axis('off')
    plt.tight_layout()

    try:
        while time.time() - start_time < duration_seconds:
            action = agent.select_action(state)
            state, _, done, truncated, _ = env.step(action)

            # Efficient frame update
            img.set_array(env.render())
            ipythondisplay.clear_output(wait=True)
            ipythondisplay.display(plt.gcf())
            frame_count += 1

            if done or truncated:
                state, _ = env.reset()

    except KeyboardInterrupt:
        pass
    finally:
        plt.close()
        print(f"\nSimulation completed! Ran for {time.time()-start_time:.1f} seconds")
        print(f"Average FPS: {frame_count/(time.time()-start_time):.1f}")

# ================== MAIN EXECUTION ==================
print("\n🚗 Starting long-duration highway simulation...")
print("Your trained model will control the car for 60 seconds")
print("Close the visualization window or interrupt the cell to stop early")

run_long_simulation(agent, env, duration_seconds=60)

env.close()
