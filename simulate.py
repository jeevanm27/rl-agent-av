import gymnasium as gym
import highway_env
import numpy as np
import tensorflow as tf
from dbsac_highway import DBSAC
import time

def simulate_trained_agent(model_path: str, num_episodes: int = 5, render_mode: str = 'human'):
    """
    Simulate the trained agent in the highway environment with visualization.
    
    Args:
        model_path: Path to the directory containing trained model weights
        num_episodes: Number of episodes to simulate
        render_mode: Rendering mode ('human' for window display, 'rgb_array' for array output)
    """
    # Create and configure the environment with rendering
    env = gym.make('highway-v0', render_mode=render_mode, config={
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
        'screen_width': 1200,  # Larger window
        'screen_height': 600,  # Larger window
        'centering_position': [0.3, 0.5],  # Center view on the ego vehicle
        'scaling': 5.5,  # Zoomed in view
        'show_trajectories': True,  # Show trajectory predictions
        'render_agent': True,  # Render the ego vehicle
    })
    
    # Get state and action dimensions
    state_dim = env.observation_space.shape
    action_dim = env.action_space.n
    
    # Initialize the agent
    agent = DBSAC(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[256, 256]
    )
    
    # Load trained weights
    try:
        agent.load_models(model_path)
        print("Successfully loaded model weights from", model_path)
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return
    
    print("\nStarting simulation...")
    print("Controls:")
    print("- Press 'r' to reset the environment")
    print("- Press 'q' to quit")
    print("- Press SPACE to pause/resume")
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        
        while not done:
            # Get action from the trained agent
            action = agent.select_action(state, evaluate=True)
            
            # Take action in the environment
            next_state, reward, done, truncated, info = env.step(action)
            
            # Render the environment
            env.render()
            
            # Update for next step
            state = next_state
            episode_reward += reward
            step += 1
            
            # Add a small delay to make visualization easier to follow
            time.sleep(0.1)
            
            if truncated:
                break
            
        print(f"Episode {episode + 1} finished with reward: {episode_reward:.2f} in {step} steps")
    
    env.close()
    print("\nSimulation completed!")

if __name__ == "__main__":
    # Specify the path to your trained model weights
    model_path = "./models"
    
    # Run the simulation
    simulate_trained_agent(model_path, num_episodes=5) 
