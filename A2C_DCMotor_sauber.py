# import libraries
import tensorflow as tf
from keras.layers import Input, Dense, Lambda
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os

#float type default
tf.keras.backend.set_floatx('float64')

path = "D:/Benutzer/alerch/sciebo/Labor_Forschung/PROJ_RMA_RheinMetall/Ergebnisse_Predator/"

#hyperparameters
gamma = 0.5            #discount factor
actor_lr = 0.00004      #actor learning rate
critic_lr = 0.0001      #critic learning rate
sample_steps = 100      #steps per episode and batch size
max_episodes = 2000    #number of episodes
delta_T = 0.001          #for DLTI
c = -0.95               #factor reward function

class actor:
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
        mu, std = self.model.predict(state, verbose=0)
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

class dcmotor:
    def __init__(self):
        self.model_dlti = self.create_dlti()

    def create_dlti(self):
        R, L, K_M, b, J = 1.93, 0.0062, 0.05247, 0.00000597, 0.00000676  # DC motor parameters
        K_E = K_M

        # state space matrices of DC motor
        A = np.array([[-(R / L), -(K_E / L)], [(K_M / J), -(b / J)]])
        B = np.array([[(1 / L), 0], [0, -(1 / J)]])
        C = np.array([0, 1])
        D = np.array([0, 0])

        return signal.cont2discrete((A, B, C, D), delta_T)

    def next_state(self, omega, action):
        # create input vector with action
        U = np.empty(shape=(sample_steps, 2), dtype=float)
        for k in range(sample_steps):
            U[k] = np.array([action[0], 0.0])

        t_out, y_out, x_out = signal.dlsim(self.model_dlti, U, t=None, x0=np.array([0, omega]))

        return y_out[1]         #return next omega


class agent:
    def __init__(self):
        self.state_dim = 2
        self.action_dim = 1
        self.action_bound = 42.0
        self.std_bound = [1e-5, 1.0]
        self.omega_bound = 1250.0 # 1/s

        self.actor = actor(self.state_dim, self.action_dim, self.action_bound, self.std_bound)
        self.critic = critic(self.state_dim)
        self.dcmotor = dcmotor()

    def td_target(self, reward, next_state, k):
        if k==(sample_steps-1):
            return reward
        v_value = self.critic.model.predict(np.reshape(next_state, [1, self.state_dim]), verbose=0)
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
                if k == 0:
                    omega_target = 500.0 / self.omega_bound
                    delta_omega = omega_target - next_omega / self.omega_bound
                    state = (delta_omega, omega_target)

                #change between target values in one episode
                # if k == 0:
                #     omega_target = 10.0 / 358.0
                #     delta_omega = omega_target-next_omega/358.0
                #     state = (delta_omega, omega_target)
                # elif k == 29:
                #     omega_target = 50.0 / 358.0
                #     delta_omega = omega_target-(next_omega[0]/358.0)
                #     state = (delta_omega, omega_target)
                # elif k == 59:
                #     omega_target = 100.0 / 358.0
                #     delta_omega = omega_target-(next_omega[0]/358.0)
                #     state = (delta_omega, omega_target)
                # elif k == 89:
                #     omega_target = 150.0 / 358.0
                #     delta_omega = omega_target-(next_omega[0]/358.0)
                #     state = (delta_omega, omega_target)
                # elif k == 119:
                #     omega_target = 200.0 / 358.0
                #     delta_omega = omega_target-(next_omega[0]/358.0)
                #     state = (delta_omega, omega_target)
                # elif k == 149:
                #     omega_target = 250.0 / 358.0
                #     delta_omega = omega_target-(next_omega[0]/358.0)
                #     state = (delta_omega, omega_target)
                # elif k == 179:
                #     omega_target = 300.0 / 358.0
                #     delta_omega = omega_target-(next_omega[0]/358.0)
                #     state = (delta_omega, omega_target)
                # elif k == 209:
                #     omega_target = 358.0 / 358.0
                #     delta_omega = omega_target-(next_omega[0]/358.0)
                #     state = (delta_omega, omega_target)

                #reversed normalisation for DC Motor
                action = self.actor.get_action(state) * self.action_bound
                #print(action)
                action = np.clip(action, -self.action_bound, self.action_bound) # undone scaling
                omega = (omega_target-state[0]) * self.omega_bound # undone scaling
                next_omega = self.dcmotor.next_state(omega, action)

                #normalisation for RL-Agent and reward calculation
                delta_omega_next = omega_target - (next_omega / self.omega_bound)
                reward = c * (abs(delta_omega_next))
                action = action/self.action_bound
                omega_target = np.clip(omega_target, 0.0, 1.0) # unnecessary?
                next_state = (delta_omega_next[0], omega_target)

                state = np.reshape(state, [1, self.state_dim])
                action = np.reshape(action, [1, self.action_dim])
                next_state = np.reshape(next_state, [1, self.state_dim])
                reward = np.reshape(reward, [1, 1])

                td_target = self.td_target(reward, next_state, k)
                advantage = self.advantage(td_target, self.critic.model.predict(state, verbose=0))

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

        np.savetxt(os.path.join(path, "results.csv"), np.array(list(zip(ep_reward_batch, actor_loss_batch, critic_loss_batch))), fmt='%.8f', delimiter=";")

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
        fig.suptitle('TotEpisodes={}, Gamma={}, SampleSteps={}, ActorLR={}, CriticLR={}'.format(ep, gamma, sample_steps, actor_lr, critic_lr), fontsize=10)
        plt.savefig(os.path.join(path, 'results.png'), dpi=500)
        #plt.show()

        self.actor.model.save(os.path.join(path, 'actor_model.keras'))

agent().train()