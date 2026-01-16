# Modeling ICU Treatment Optimization Using Actor-Critic Reinforcement Learning

Title: Actor-Critic Based Sepsis Treatment Optimization<br>

## Scenario: 
You are working as a data scientist in a medical AI research team. Your objective is to train a reinforcement learning agent that can suggest optimal treatment strategies for sepsis patients in the ICU. Sepsis is a life-threatening condition, and its management requires timely interventions based on real-time vitals and clinical decisions. ICU patients are monitored continuously, and treatment decisions (actions) are logged over time.<br>
## Objective
The goal of this assignment is to model the ICU treatment process using Reinforcement Learning, specifically the Actor-Critic method. The agent should learn an optimal intervention policy from historical ICU data. Each patient's ICU stay is treated as an episode consisting of time-stamped clinical observations and treatments.<br>
Your tasks:<br>
&emsp;1. Model the ICU treatment process as a Reinforcement Learning (RL) environment.<br>
&emsp;2. Train an Actor-Critic agent to suggest medical interventions based on the patient’s current state (vitals and demographics).<br>
Dataset:<br>
Use the dataset provided in the following link:
https://drive.google.com/file/d/1UPsOhUvyrsrC59ilXsvHwGZhzm7Yk01w/view?usp=sharing<br>
Features:<br>
&emsp;● Vitals: mean_bp, spo2, resp_rate<br>
&emsp;● Demographics: age, gender<br>
&emsp;● Action: Medical intervention (e.g., "Vancomycin", "NaCl 0.9%", or NO_ACTION)<br>
&emsp;● Identifiers: timestamp, subject_id, hadm_id, icustay_id<br>
## Environment 
Setup (RL Formulation)<br>
State Space<br>
Each state vector consists of: mean_bp (Mean Blood Pressure) , spo2 (Oxygen Saturation), resp_rate (Respiratory Rate), age, One-hot encoded gender. <br>
## Action Space
&emsp;● The agent selects one discrete action from 99 possible medical interventions (e.g., Vancomycin, Fentanyl, PO Intake, etc.<br>
&emsp;● You should integer encode or one-hot encode these interventions.<br>
## Reward
At each time step, the agent receives a reward based on how close the patient's vitals are to clinically normal ranges. The reward encourages the agent to take actions that stabilize the patient's vital signs:<br>
Rewardt = − ((MBPt −90)2+(SpO2t −98)2+(RRt −16)2)<br>
## Explanation:
&emsp;● MBP (mean_bp): Target = 90 mmHg<br>
&emsp;● SpO₂ (spo2): Target = 98%<br>
&emsp;● RR (resp_rate): Target = 16 breaths/min<br>
Each term penalizes the squared deviation from the healthy target. The smaller the difference, the higher (less negative) the reward.<br>
## Example:
Suppose at time t, the vitals are:<br>
&emsp;● MBP = 88<br>
&emsp;● SpO₂ = 97<br>
&emsp;● RR = 20<br>
### Then the reward is:
Rewardt = − [(88−90)2+(97−98)2+(20−16)2] = − (4+1+16)= −21<br>
A lower (more negative) reward indicates worse vitals, guiding the agent to learn actions that minimize this penalty.
## Episode termination
An episode ends when the ICU stay ends. To define this:<br>
&emsp;1. Group the data by subject_id, hadm_id, icustay_id → Each group represents one ICU stay = one episode.<br>
&emsp;2. Sort each group by timestamp → Ensure the time progression is correct.<br>
&emsp;3. For each time step in a group (i.e., each row). Check if it is the last row in that group. → If yes, then mark done = True (end of episode). → If no, then done = False (continue episode).<br>

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

1. Objective of the Project

The primary objective of this assignment is to model the Intensive Care Unit (ICU) treatment process as a reinforcement learning problem and to learn an optimal medical intervention policy using the Actor-Critic method. Each patient’s ICU stay is treated as a single episode composed of time-stamped clinical observations and administered treatments. The reinforcement learning agent is trained on historical ICU data to recommend medical interventions based on the patient’s current physiological state and demographic information. The ultimate goal is to minimize patient instability by learning a policy that keeps vital signs as close as possible to clinically normal ranges over the duration of the ICU stay.

2. Problem Formulation as a Reinforcement Learning Task

The ICU treatment process is formulated as a sequential decision-making problem under uncertainty. At each time step, the agent observes the patient’s current state, which includes vital signs and demographic features, and selects a medical intervention from a large discrete action space. The environment then transitions to the next time step, producing a reward based on how the patient’s vitals respond. Because patient trajectories are sequential and outcomes depend on both current and past decisions, reinforcement learning provides a natural framework for learning long-term treatment strategies rather than short-sighted, greedy decisions.

3. Dataset Description and Clinical Features

The dataset used in this project consists of historical ICU records containing patient vitals, demographic information, administered treatments, and identifiers for ICU stays. Vital signs include mean blood pressure, oxygen saturation, and respiratory rate, which are critical indicators of patient stability. Demographic features such as age and gender are included to provide contextual information that may influence treatment decisions. Each row represents a time-stamped snapshot of a patient’s ICU stay, and identifiers such as subject ID, hospital admission ID, and ICU stay ID are used to group observations into complete episodes.

4. State Space Representation

Each state in the reinforcement learning environment is represented as a numerical feature vector composed of the patient’s vital signs and demographic attributes. Specifically, the state includes mean blood pressure, oxygen saturation, respiratory rate, patient age, and one-hot encoded gender. These features collectively describe the patient’s physiological condition at a given time step. To ensure stable and efficient neural network training, all state features are normalized to a common range using min-max scaling. This normalization prevents features with larger magnitudes from dominating the learning process.

5. Action Space Definition

The action space consists of 99 discrete medical interventions, including various medications, fluid administrations, and a no-action option. Each intervention is integer encoded to allow the agent to select actions using a categorical probability distribution. This discrete action formulation reflects the real-world clinical setting, where clinicians choose from a finite set of treatment options at each decision point. The large action space increases the complexity of the learning problem and highlights the need for function approximation through deep neural networks.

6. Reward Function Design

The reward function is designed to reflect clinical stability by penalizing deviations of vital signs from healthy target values. At each time step, the agent receives a negative reward proportional to the squared difference between observed vitals and predefined clinical targets for mean blood pressure, oxygen saturation, and respiratory rate. This formulation encourages the agent to take actions that move the patient’s condition closer to physiological norms. Squared penalties are used to disproportionately penalize large deviations, emphasizing the importance of avoiding extreme instability. As a result, higher (less negative) rewards correspond to healthier patient states.

7. Episode Definition and Termination

Each episode corresponds to a single ICU stay and is defined by grouping data using subject ID, hospital admission ID, and ICU stay ID. Observations within each episode are sorted chronologically using timestamps to preserve temporal order. An episode terminates when the last recorded time step for a given ICU stay is reached. This episodic structure aligns naturally with patient care trajectories and allows the agent to learn treatment policies that consider the full course of an ICU admission rather than isolated interventions.

8. Data Preprocessing and Feature Engineering

Before training, the dataset undergoes extensive preprocessing to ensure temporal consistency and numerical stability. Missing values are handled using forward filling, which is appropriate for time-series clinical data where measurements are often recorded intermittently. Categorical variables such as gender are one-hot encoded, while medical interventions are integer encoded to form a discrete action space. State features are normalized to improve convergence during training. Additionally, a terminal flag is computed to indicate the end of each ICU episode, enabling proper episode segmentation during training.

9. Custom Environment Design (SepsisTreatmentEnv)

A custom reinforcement learning environment is implemented to wrap the static ICU dataset and mimic the behavior of an interactive environment. The environment exposes step and reset methods similar to the OpenAI Gym interface. Rather than generating transitions dynamically, the environment replays historical patient trajectories, returning state-action-reward-next-state tuples sequentially. This offline reinforcement learning setup allows the agent to learn from previously collected clinical data while maintaining a clear separation between environment dynamics and learning logic.

10. Reward Computation from Unscaled Clinical Data

To ensure clinical interpretability, rewards are computed using the original, unscaled vital sign values. The reward function is applied to the raw dataset before normalization, ensuring that deviations are measured in meaningful physiological units. These rewards are then attached to the normalized dataset used for training. This separation preserves clinical realism while still benefiting from normalized inputs during neural network optimization.

11. Actor-Critic Algorithm Overview

The Actor-Critic method combines the strengths of policy-based and value-based reinforcement learning approaches. The actor network learns a stochastic policy that outputs a probability distribution over medical interventions, while the critic network estimates the expected value of a given patient state. By jointly learning both components, the agent can efficiently evaluate its actions and update the policy in a direction that maximizes long-term reward. This approach is particularly well-suited for problems with large action spaces and continuous state representations, such as ICU treatment optimization.

12. Neural Network Architecture

Both the actor and critic share a common neural network backbone that extracts high-level representations from the patient state features. This shared structure improves sample efficiency and ensures that both networks learn consistent feature representations. The actor head outputs a softmax distribution over possible interventions, while the critic head outputs a scalar value estimate. Separate optimizers and learning rates are used for the actor and critic to balance policy learning and value estimation.

13. Training Process and Temporal Difference Learning

Training proceeds by iterating through the dataset sequentially, treating it as a series of episodes. At each step, the agent computes the temporal difference (TD) target using the observed reward and the critic’s estimate of the next state value. The TD error, representing the advantage of the taken action, is used to update both the actor and critic networks. The critic is trained using a Huber loss for robustness to outliers, while the actor is updated to increase the probability of actions that lead to positive TD errors. This learning process enables the agent to gradually improve its treatment recommendations.

14. Offline Reinforcement Learning Considerations

Since the agent is trained on historical data, the action selected by the actor is not executed in the environment; instead, the environment replays the recorded clinician action. This offline learning setup avoids unsafe exploration but limits the agent to learning from observed behavior. Despite this limitation, the Actor-Critic framework can still extract valuable patterns and learn a policy that aligns with improved patient outcomes as defined by the reward function.

15. Policy Evaluation and Reward Trends

The learned policy is evaluated by tracking cumulative rewards over ICU episodes. Rolling averages of episode rewards reveal a clear upward trend during training, indicating that the agent is learning to recommend interventions associated with improved patient stability. Initially, rewards are highly negative, reflecting poor alignment with the clinical targets. Over time, rewards increase and approach a plateau, suggesting convergence toward a stable and effective policy.

16. Stability and Convergence Behavior

The flattening of the reward curve in later episodes indicates that the policy updates become smaller as training progresses. This behavior suggests that the agent has converged to a stable strategy that consistently minimizes vital sign deviations. The use of advantage-based updates and a separate value estimator contributes to reduced variance and improved stability compared to pure policy gradient methods.

17. Clinical Interpretation and Limitations

While the learned policy demonstrates promising behavior within the simulated environment, its real-world applicability depends heavily on the quality and completeness of the historical data. The reward function assumes that stabilizing vitals directly corresponds to better clinical outcomes, which may not always hold in complex medical settings. Additionally, offline training restricts the agent from exploring novel treatment strategies beyond those observed in the dataset.

18. Conclusion

This project presents a structured application of Actor-Critic reinforcement learning to ICU treatment optimization. By modeling patient trajectories as episodes and learning from historical data, the agent acquires a policy that recommends interventions aimed at stabilizing vital signs. The project demonstrates how reinforcement learning can be applied to healthcare decision-making while highlighting the importance of careful environment modeling, reward design, and ethical considerations. Overall, the work provides a strong foundation for further research in data-driven clinical decision support systems.
