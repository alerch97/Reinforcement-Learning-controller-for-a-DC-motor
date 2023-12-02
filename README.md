# Reinforcement-Learning-controller for a DC-motor
Topic of my bachelor thesis from 2020

## Aim
This is an approach of a control design for a DC-motor with the help of Reinforcement Learning. The controller (RL-agent) consists of an artificial neural network, which is trained by the A2C-algorithm (Advantage Actor Critic). The environment is characterized by a simple discrete LTI-system (linear time invariant) of the motor. The aim is that the RL-agent is able to pass the right voltage to the motor to reach the desired angular velocity. Regarding the stationary as well as the dynamic behaviour a brief analysis is performed.

## Environment
I decided to create a state-space representation of the DC-motor. There is the possibility to implement this kind auf physical model with SciPy. The following figure represents the simplified physical equivalent circuit diagram.

![Ersatzschaltbild](https://github.com/alerch97/Reinforcement-Learning-for-a-DC-motor/assets/152506794/0f871774-6cd9-40da-991b-2744136e2e62)

This can now be used to create the equation of the state-space representation:

![image](https://github.com/alerch97/Reinforcement-Learning-for-a-DC-motor/assets/152506794/030f460c-ab88-4425-83af-747fa4a6e0e2)

The states taken into consideration and actions made by the RL-agent are valid for a specific moment. Therefore the state-space model will be time-discretized with a fixed time step of 0.05 s. In combination with the selected motor parameters you cannot see discontinuities in step responses (here not included). To implement these considerations we create a class for the DC-motor:
```python
class dcmotor:
    def __init__(self):
        self.model_dlti = self.create_dlti()

    def create_dlti(self):
        R, L, Kn, Km, J = 0.365, 0.000161, 1.3, 0.123, 0.02
        A = np.array([[-R/L, -1.0/(2.0*3.14*Kn*L)], [Km/J, 0.0]])
        B = np.array([[1.0/L], [0.0]])
        C = np.array([0.0, 1.0])
        D = np.array([0.0])
        return signal.cont2discrete((A, B, C, D), delta_T)

    def next_state(self, omega, action):
        #create input vector with action
        U = []
        for f in range(sample_steps):
            U.append(action)
        t_out, y_out, x_out = signal.dlsim(self.model_dlti, U, None, omega)
        return y_out[1]         #return next omega
```
By calling this class, the DC-motor gets initialised as a DLTI-system.

## Realisation Idea
In the following figure you can see the overall implementation idea as well as the control structure and interfaces between the different needed parts. It is important to note that after successful training, the RL-agent only consists of the actor (inclusively pre- post-processing tasks).

![Kontrollstruktur_englisch](https://github.com/alerch97/Reinforcement-Learning-controller-for-a-DC-motor/assets/152506794/3491898d-25fb-44f8-9c09-809fc90dd47b)

The interfaces of the DC-motor are the voltage $u$ as an input and the current angular velocity omega as an output. In this context, the voltage $u$ represents the RL-agent's action. Following this action, the DC-motor responds in the next discrete time step with the current angular velocity at time step $t+1$, which is then fed back to the RL-agent. This feedback loop informs the RL-agent about the consequences of its previous action. To enhance the RL-agent's ability to achieve desired outcomes, not only the current angular velocity $omega_t$ is provided, but also of course the target angular velocity $omega_{target}$. This allows the RL-agent to work towards reaching the specified target value in subsequent steps. However, for the agent, the control difference $delta_{omega}$ is derived from these two values and fed back to the system. Therefore the State vector $S_t$ consists of $delta_{omega}$ and $omega_{target}$. In addition, the elements of the state vector are scaled when they enter the RL-agent so that they are in the range $[-1, 1]$ (Min-Max-scaling). Negative control differences and negative voltages can also be represented. Of course this scaling must be reversed for the action when leaving the RL-agent. The scaling task is a part of the RL-agent.<br>
Furthermore, to train the RL-agent using the A2C-algorithm, a reward function must be defined. The function follows specific criteria:
- maximal reward when the control difference is $0 s^{−1}$,
- a maximum reward value of $0$,
- negative values for non-zero control differences and
- increasingly negative rewards for higher control differences.
          
A linear equation with a negative factor c and the absolute value of the control difference meet these requirements. The factor c determines the penalty strength for deviation from the desired angular velocity, also assigned as a hyperparameter for training success:
$$R_t \left(\Delta\omega_t\right) = c\cdot |\Delta\omega_t|.$$

Before heading to the code implementations, we will have a look on the flow chart of the RL-agent's training process. After initialising the DC-motor-model as well as the actor and critic neural networks along with hyperparameters, the training starts with episode 0. At step $k = 0$, an initial state is set and an action is chosen. This leads to a new state $S_{k+1}$ and a reward $R_{k+1}$. Subsequently, the TD-target and advantage function value are computed and stored. The step count $k$ is incremented. If the maximum step count per episode is not reached, both the current state $S_k$ and the cumulative reward $R$ are updated. The program then returns to its flow and starts again with the fourth process. If the condition $k = k_{max}$ is met, the two neural networks are updated. If the maximum episode count is not reached, the current episode count is incremented and the process restarts with the initialisation of the starting state, $k=0$ and $R=0$. If this condition is affirmed, training data and actor neural network are saved, marking the end of the training (sorry, the flow chart is still in German).

![Programmablaufplan](https://github.com/alerch97/Reinforcement-Learning-controller-for-a-DC-motor/assets/152506794/f71d5531-a4a0-49c7-8061-57ab423fd13e)

For the higher-level code structur beside the class for the motor model, I specify further classes for the actor, critic and the agent. Moreover, the initialisation of different hyperparameters takes place in the beginning. Calling the function agent().train() the execution of the training process starts.
```python
# import libraries
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

#float type default
tf.keras.backend.set_floatx('float64')

#hyperparameters
gamma = 0.5            #discount factor
actor_lr = 0.00006      #actor learning rate
critic_lr = 0.0002      #critic learning rate
sample_steps = 240     #steps per episode and batch size
max_episodes = 2000    #number of episodes
delta_T = 0.05          #for DLTI
c = -0.95               #factor reward function

class actor:...

class critic:...

class dcmotor:...

class agent:...

agent().train()
```









