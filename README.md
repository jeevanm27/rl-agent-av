# 🚗 SafeLane: Adaptive Safety-Driven RL Agent for Lane Changing

![SafeLane Demo](output.mp4)

> **Authors**: Jeevan M, M. Sridevi, Darshan R  
> 📄 **Paper**: *Adaptive Safety-Driven Deep Q-Network for Autonomous Lane Changing in Highway Environments*  
> 🧪 Simulation: `highway-env` | 🧠 Model: Deep Q-Network (DQN) with Adaptive Rewards

---

## 📌 Abstract

SafeLane is a Deep Reinforcement Learning-based autonomous lane-changing agent designed to balance **efficiency** and **safety** in highway scenarios. Unlike standard DQNs that use fixed reward functions, SafeLane employs an **adaptive reward shaping mechanism** that penalizes unsafe behaviors like tailgating and abrupt lane switches, while encouraging smooth, context-aware decisions.

✅ Built using [`highway-env`](https://github.com/eleurent/highway-env)  
✅ Implements custom DQN with safety-aware rewards  
✅ Trained and evaluated over 5000 episodes in dynamic traffic conditions  
✅ Final agent shows reduced collisions and better generalization  
✅ Output demo video included (`output.mp4`)  
✅ We can continue training from the saved model and buffer by extending the number of episodes

---

## 📁 Project Structure

.
├── final_model.py               # Training logic and DQN model
├── final_simulation.py          # Simulation script for inference
├── output.mp4                   # Sample simulation output
├── research paper.docx          # Research paper with methods and results
├── models/                      # Saved model and training data
│   ├── q_network.weights.h5
│   ├── target_network.weights.h5
│   ├── replay_buffer.pkl
│   └── training_metadata.json
└── README.md                    # This file

🧠 Highlights
Adaptive Reward Function:

r_t = R_progress + R_safety + R_comfort

Dynamic feedback based on:

Nearby vehicle proximity

Lane-change smoothness

Rule violations and density

Deep Q-Network Setup:

Multi-Layer Perceptron (MLP)

Double DQN to reduce overestimation

Prioritized Experience Replay

Epsilon-Greedy exploration with decay

Soft target updates for training stability

Environment:

highway-v0 from highway-env

Discrete action space: ["stay", "left", "right"]

Randomized traffic conditions with realistic driving dynamics

🚀 Getting Started

🔧 Installation

git clone https://github.com/jeevanm27/rl-agent-av.git
cd rl-agent-av

python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

pip install tensorflow>=2.13.0 numpy>=1.24.0 gymnasium>=0.29.0 highway-env>=1.8.0

🏁 Training (from scratch or resume)

python final_model.py

✅ To resume training, ensure the following files exist in the models/ directory:

q_network.weights.h5

target_network.weights.h5

replay_buffer.pkl

training_metadata.json

🎮 Run Simulation (evaluate agent)
python final_simulation.py

📊 Training Results
Episode Range	Mean Reward
1000	54
2000	108
3000	126
4000	144
5000	150

🧪 Ablation Study Summary
Component Removed	Observed Effect
Adaptive Reward	Increased collisions, unstable policy
Prioritized Replay	Slower convergence, less stability
Multi-objective Reward	Over-focus on speed/lane, less safe

📎 Research Paper
📄 research paper.docx — contains methodology, literature review, equations, training graphs, and references.

💾 Checkpoints
📂 models/ directory contains all necessary files to resume training:

q_network.weights.h5 — main Q-network

target_network.weights.h5 — target network

replay_buffer.pkl — experience replay data

training_metadata.json — training progress, rewards


🧠 Citation
If you use this project in your research, please cite:
@article{jeevan2025safelane,
  title={Adaptive Safety-Driven Deep Q-Network for Autonomous Lane Changing in Highway Environments},
  author={Jeevan M and M. Sridevi and Darshan R},
  journal={Under Review},
  year={2025}
}
📬 Contact
📧 Email: jeevan231227@gmail.com
🔗 GitHub: @jeevanm27
