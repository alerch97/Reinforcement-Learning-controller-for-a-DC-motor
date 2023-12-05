# Reinforcement-Learning-controller for a DC-motor
Topic of my bachelor thesis from 2020

## Aim
This is an approach of a control design for a DC-motor with the help of Reinforcement Learning. The controller (RL-agent) consists of an artificial neural network, which is trained by the A2C-algorithm (Advantage Actor Critic). The environment is characterized by a simple discrete LTI-system (linear time invariant) of the motor. The aim is that the RL-agent is able to pass the right voltage to the motor to reach the desired angular velocity. Regarding the stationary as well as the dynamic behaviour a brief analysis is performed.

## A2C-algorithm
The Advantage-Actor-Critic-algorithm combines the advantages of Temporal Difference Learning (TD-Learning) and the Policy Gradient method. Here the actor learns the strategy using the Policy Gradient method and accordingly executes new actions during training. In contrast, the critic evaluates the current state through the value function forced by the actor's action. Similar to the actor, the critic learns the value function through TD-Learning. The goal, identical to TD-Learning, is to apply the Actor-critic method to continuous problems. For a continuouse action range it is a common idea to choose a gaussian probability density function for the actor's strategy $\pi$:
$$\pi(a|s,\theta) = \dfrac{1}{\sigma(s,\theta)\sqrt(2\pi)} \exp(-\dfrac{(a-\mu(s,\theta))^2}{2\sigma(s,\theta)^2}).$$
The advantage function $A_t$ is characterised by the adjustment with the baseline $v(S_t,w_t)$ to bound the variance for the weighting of gradients ($w$ are the weights of the critic's neural network):
$$A_t = R_{t+1} + \gamma v(S_{t+1},w_t) - v(S_t,w_t).$$
This function is necessary for updating the weights $\theta$ of the actor's neural network (with a specific learning rate $\alpha$):
$$\theta_{t+1} = \theta_{t} + \alpha A_t \nabla_{\theta} [ln(\pi (A_t|S_t,\theta_t))].$$
For updating the weights of the critic, we have to implement the following equation (with a specific learning rate $\alpha$):
$$w_{t+1} = w_{t} + \alpha A_t \nabla_{w} [v (S_t,w_t)].$$

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
## Agent
By calling the agent class, the actor critic and DC-motor get initialised. The dimensions of the state and action vectors are set. The action bound is fixed to 50 volts. For the random selection of an action the standard deviation is also bounded. The train function starts with initialising empty lists. For the start of an episode the motor start with an angular velocity of $\omega=0$. With the if else query during an episode, a continouse change of the target angular velocity is performed (can be better realised maybe with methods of Desing of Experiments (DoE)). Based on the current state, the recommended action of the actor is calculated and gets fed to the motor for determining the next state. With the new reward the values of TD-target and the advantage function are calculated. If $k$ reaches the maximum step count of 240, an update of the actor's and critic's neural networks is performed. After finishing the training process, different plots are displayed (losses and reward).
```python
class agent:
    def __init__(self):
        self.state_dim = 2
        self.action_dim = 1
        self.action_bound = 50.0
        self.std_bound = [1e-5, 1.0]

        self.actor = actor(self.state_dim, self.action_dim, self.action_bound, self.std_bound)
        self.critic = critic(self.state_dim)
        self.dcmotor = dcmotor()

    def td_target(self, reward, next_state, k):
        if k==(sample_steps-1):
            return reward
        v_value = self.critic.model.predict(np.reshape(next_state, [1, self.state_dim]))
        return np.reshape(reward + gamma * v_value[0], [1, 1])

    def advantage(self, td_target, baseline):
        return td_target - baseline

    def list_to_batch(self, list):
        batch = list[0]
        for elem in list[1:]:
            batch = np.append(batch, elem, axis=0)
        return batch

    def train(self):
        ep_batch = []
        ep_reward_batch = []
        actor_loss_batch = []
        critic_loss_batch = []
        for ep in range(max_episodes):
            state_batch = []
            action_batch = []
            td_target_batch = []
            advantage_batch = []
            ep_batch.append(ep)
            episode_reward = 0
            next_omega = 0.0     #initial state for every new episode

            for k in range(sample_steps):
                #change between target values in one episode
                if k == 0:
                    omega_target = 10.0 / 358.0
                    delta_omega = omega_target-next_omega/358.0
                    state = (delta_omega, omega_target)
                elif k == 29:
                    omega_target = 50.0 / 358.0
                    delta_omega = omega_target-(next_omega[0]/358.0)
                    state = (delta_omega, omega_target)
                elif k == 59:
                    ...

                #reversed normalisation for DC Motor
                action = self.actor.get_action(state) * 50.0
                print(action)
                action = np.clip(action, -self.action_bound, self.action_bound)
                omega = (omega_target-state[0]) * 358.0
                next_omega = self.dcmotor.next_state(omega, action)

                #normalisation for RL-Agent and reward calculation
                delta_omega_next = omega_target - (next_omega / 358.0)
                reward = c * (abs(delta_omega_next))
                action = action/50.0
                omega_target = np.clip(omega_target, 0.0, 1.0)
                next_state = (delta_omega_next[0], omega_target)

                state = np.reshape(state, [1, self.state_dim])
                action = np.reshape(action, [1, self.action_dim])
                next_state = np.reshape(next_state, [1, self.state_dim])
                reward = np.reshape(reward, [1, 1])

                td_target = self.td_target(reward, next_state, k)
                advantage = self.advantage(td_target, self.critic.model.predict(state))

                state_batch.append(state)
                action_batch.append(action)
                td_target_batch.append(td_target)
                advantage_batch.append(advantage)

                if len(state_batch) >= sample_steps:
                    states = self.list_to_batch(state_batch)
                    actions = self.list_to_batch(action_batch)
                    td_targets = self.list_to_batch(td_target_batch)
                    advantages = self.list_to_batch(advantage_batch)

                    actor_loss = self.actor.train(states, actions, advantages)
                    critic_loss = self.critic.train(states, td_targets)

                episode_reward += reward[0][0]
                state = next_state[0]


            actor_loss_proto = tf.make_tensor_proto(actor_loss)
            critic_loss_proto = tf.make_tensor_proto(critic_loss)
            actor_loss_array = tf.make_ndarray(actor_loss_proto)
            critic_loss_array = tf.make_ndarray(critic_loss_proto)
            actor_loss_batch.append(actor_loss_array)
            critic_loss_batch.append(critic_loss_array)
            ep_reward_batch.append(episode_reward)
            print('EP{} EpisodeReward={} ActorLoss={} CriticLoss={}'.format(ep, episode_reward, actor_loss_array, critic_loss_array))

        # plot episode reward and losses over episodes
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        ax1.plot(ep_batch, ep_reward_batch)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Episode reward')
        ax2.plot(ep_batch, actor_loss_batch)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Actor loss')
        ax3.plot(ep_batch, critic_loss_batch)
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Critic loss')
        fig.suptitle('TotEpisodes={}, Gamma={}, SampleSteps={}, ActorLR={}, CriticLR={}'.format(ep, gamma, sample_steps, actor_lr, critic_lr))
        plt.show()

        self.actor.model.save_weights('weights/actor_weights.h5')
```

## Actor
In the beginning, the actor's neural network with two hidden layers (each with 100 neurons) is initialised. The output of the actor is a gaussian distribution with a mean $\mu$ and a standard deviation $\sigma$. Initially, $\sigma$ is quite big, which leads to actions being further away from the mean to guarantee an exploration of the whole action range. The softplus activation function guarantees the range $[0,1]$ for $\sigma$. The selection of the tanh activation function for $\mu$ is examined later. By performing a prediction for the current state, we get next action (voltage) for the motor. To perform the training via gradient descent, we have to define a compatabile loss function based on the gaussian output. Due to the characteristics of the A2C-algorithm, the loss consists of the logarithmic probability density function times the value of the advantage function (provided by the critic).

```python
def __init__(self, state_dim, action_dim, action_bound, std_bound):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.std_bound = std_bound
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(actor_lr)

    def create_model(self):
        state_input = Input((self.state_dim,))
        dense_1 = Dense(100, activation='relu')(state_input)
        dense_2 = Dense(100, activation='relu')(dense_1)
        out_mu = Dense(self.action_dim, activation='tanh')(dense_2)
        std_output = Dense(self.action_dim, activation='softplus')(dense_2)
        mu_output = Lambda(lambda x: x * 1.0)(out_mu)
        return tf.keras.models.Model(state_input, [mu_output, std_output])

    def get_action(self, state):
        state = np.reshape(state, [1, self.state_dim])
        mu, std = self.model.predict(state)
        mu, std = mu[0], std[0]
        return (np.random.normal(mu, std, size=self.action_dim))

    def log_pdf(self, mu, std, actions):
        std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])
        var = std ** 2
        log_policy_pdf = -0.5 * (actions - mu) ** 2 / \
                         var - 0.5 * tf.math.log(var * 2 * np.pi)
        return tf.reduce_sum(log_policy_pdf, 1, keepdims=True)

    def compute_loss(self, mu, std, actions, advantages):
        log_policy_pdf = self.log_pdf(mu, std, actions)
        loss_policy = log_policy_pdf * advantages
        return tf.reduce_mean(-loss_policy)

    def train(self, states, actions, advantages):
        with tf.GradientTape() as tape:
            mu, std = self.model(states, training=True)
            loss = self.compute_loss(mu, std, actions, advantages)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss
```

## Critic
In the beginning, the critic's neural network with two hidden layers (each with 100 neurons) is initialised. The output of the critic is a linear activation function because the value of the value function can be whatever negative or positive. The loss function is the mean squared error between the by the critic's neural network predicted value function and the TD-target. Here the neural networks gets updated regarding the best guess (TD-target).

```python
class critic:
    def __init__(self, state_dim):
        self.state_dim = state_dim
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(critic_lr)

    def create_model(self):
        return tf.keras.Sequential([
            Input((self.state_dim,)),
            Dense(100, activation='relu'),
            Dense(100, activation='relu'),
            Dense(1, activation='linear')
        ])

    def compute_loss(self, v_pred, td_targets):
        mse = tf.keras.losses.MeanSquaredError()
        return mse(td_targets, v_pred)                  #MSE of advantage function

    def train(self, states, td_targets):
        with tf.GradientTape() as tape:
            v_pred = self.model(states, training=True)
            assert v_pred.shape == td_targets.shape
            loss = self.compute_loss(v_pred, tf.stop_gradient(td_targets))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss
```

## Choosing the right hyperparameters and network topologies
The right choice of hyperparameters and the topologies of the neural networks (for the actor and thr critic) is more or less a trial-and-error-process. Nevertheless, we should think about instead of being in the dark. Important to mention: we cannot have a look at the validation loss like in supervised training because we don't know the truth! A successful RL-agent depends a lot on the specified reward function!

### Activation function for the actor's $\mu$
The two hypothetical acitvation functions (linear and TanH) for the output neuron $\mu$ of the actor get checked via a first training for 500 episodes. In the following figure you can see a significant difference. The actor loss and the cumulative reward start to decrease with passing 350 episodes and 400 episodes respectively. The TanH activation function seems to work here better.

![image](https://github.com/alerch97/Reinforcement-Learning-controller-for-a-DC-motor/assets/152506794/0f09bd52-d0c3-447a-bc27-c6ae6b02a309)

### Neurons and hidden layers
In general, I would highly recommend an intensive investigation of different topologies for both networks as well as diffenrent combinations of both networks to check for each other's impact. Here I have run just a couple of short trainings. The charts below show that the use of 100 neurons strives for faster and clearer convergence. The neural networks with the fewest neurons show a very slow convergence in comparison. In addition, a greater spread of values can be recognised in their progressions. The use of 50 neurons shows a somewhat clearer tendency to converge, but requires more time (especially at the beginning of training). This tendency is visible after approx. 100 to 150 episodes. Similarly, a greater spread of the progresses can be recognised at the beginning. For this reason, the architecture with 100 neurons per hidden layer is chosen. There may be a risk that the RL-agent will not be able to generalise the strategy to other angular velocities (missing exploration and / or overfitting). This needs to be checked after training. If this is the case, the number of neurons needs to be adjusted.

![image](https://github.com/alerch97/Reinforcement-Learning-controller-for-a-DC-motor/assets/152506794/1a66a03d-fd29-4e8d-8c38-2e551c1c2ed4)


### Expansion of the action range

### Reducing factor $\gamma$

## Training




