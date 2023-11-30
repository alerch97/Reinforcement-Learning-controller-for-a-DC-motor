# Reinforcement-Learning-controller for a DC-motor
Topic of my bachelor thesis from 2020

## Aim
This is an approach of a control design for a DC-motor with the help of Reinforcement Learning. The controller (RL-agent) consists of an artificial neural network, which is trained by the A2C-algorithm (Advantage Actor Critic). The environment is characterized by a simple discrete LTI-system of the motor. The aim is that the RL-agent is able to pass the right voltage to the motor to reach the desired angular velocity. Regarding the stationary as well as the dynamic behaviour a brief analysis is performed.

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
