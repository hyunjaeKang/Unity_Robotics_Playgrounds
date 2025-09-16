import numpy as np
import random
import copy
import datetime
import platform
import torch
import os
import torch.nn.functional as F
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from pathlib import Path
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel\
                             import EngineConfigurationChannel
from mlagents.trainers.demo_loader import demo_to_buffer
from mlagents.trainers.buffer import BufferKey, ObservationKeyPrefix


# Global Setting
cur_dir = os.path.dirname(__file__)
# env_dir = os.path.abspath(os.path.join(cur_dir, "..", "Unity2023_Envs"))
env_dir = os.path.abspath(os.path.join(cur_dir, "..", "Unity6000_Envs"))
test_dir = os.path.abspath(os.path.join(cur_dir, "temp", "Unity6000_Test"))

# Pytorch Device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# Unity Enviroment
game = "Kart_Country"
os_name = platform.system()

if os_name == 'Linux':
    env_name = os.path.join(env_dir, f"{game}_{os_name}.x86_64")
elif os_name == 'Darwin':
    env_name = os.path.join(env_dir, f"{game}_{os_name}.app")


# Seting parameters for DDPG Network
state_size = 12 * 4
action_size = 1

load_model = False
train_mode = True

batch_size = 128
learning_rate = 3e-4
discount_factor = 0.9

train_epoch = 500
test_step = 10000

print_interval = 10
save_interval = 100

# Demonstration
demo_path = os.path.abspath(os.path.join(cur_dir, "demo", "KartAgent.demo"))

# NN model : Save and Load
date_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
save_path = os.path.join(test_dir, f"saved_models/{game}_BC_{date_time}")
Path(save_path).mkdir(parents=True, exist_ok=True)
load_path = os.path.join(test_dir, f"saved_models/Kart_BC_20250411165212") # Need to update


# Actor class -> Behavioral Cloning Actor class
class Actor(torch.nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = torch.nn.Linear(state_size, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.mu = torch.nn.Linear(128, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.mu(x))

# BCAgent class
class BCAgent():
    def __init__(self):
        self.actor = Actor().to(device)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.writer = SummaryWriter(save_path)

        if load_model == True:
            print(f"... Load Model from {load_path}/ckpt ...")
            checkpoint = torch.load(load_path+'/ckpt', map_location=device)
            self.actor.load_state_dict(checkpoint["actor"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])

    #
    def get_action(self, state, training=False):
        #  네트워크 모드 설정
        self.actor.train(training)
        action = self.actor(torch.FloatTensor(state).to(device)).cpu().detach().numpy()
        return action

    #
    def train_model(self, state, action):
        losses = []

        rand_idx = torch.randperm(len(state))
        for iter in range(int(np.ceil(len(state)/batch_size))):
            _state = state[rand_idx[iter*batch_size: (iter+1)*batch_size]]
            _action = action[rand_idx[iter*batch_size: (iter+1)*batch_size]]

            action_pred = self.actor(_state)
            loss = F.mse_loss(_action, action_pred).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

        return np.mean(losses)

    #
    def save_model(self):
        print(f"... Save Model to {save_path}/ckpt ...")
        torch.save({
            "actor" : self.actor.state_dict(),
            "optimizer" : self.optimizer.state_dict(),
        }, save_path+'/ckpt')


    def write_summray(self, loss, epoch):
        self.writer.add_scalar("model/loss", loss, epoch)
# Behavior cloning
if __name__ == '__main__':

    agent = BCAgent()

    if train_mode:
        # get the information of demo
        behavior_spec, demo_buffer = demo_to_buffer(demo_path, 1)
        print(demo_buffer._fields.keys())

        demo_to_tensor = lambda key: torch.FloatTensor(demo_buffer[key]).to(device)
        state = demo_to_tensor((ObservationKeyPrefix.OBSERVATION, 0))
        action = demo_to_tensor(BufferKey.CONTINUOUS_ACTION)
        reward = demo_to_tensor(BufferKey.ENVIRONMENT_REWARDS)
        done = demo_to_tensor(BufferKey.DONE)

        ret = reward.clone()
        for t in reversed(range(len(ret) - 1)):
            ret[t] += (1. - done[t]) * (discount_factor * ret[t+1])

        # use the pair of (state, action) which is greater than 0
        state, action = map(lambda x: x[ret > 0], [state, action])

        losses = []
        for epoch in range(1, train_epoch+1):
            loss = agent.train_model(state, action)
            losses.append(loss)

            # record tensorboard
            if epoch % print_interval == 0:
                mean_loss = np.mean(losses)
                print(f"{epoch} Epoch / Loss: {mean_loss:.8f}" )
                agent.write_summray(mean_loss, epoch)
                losses = []

            if epoch % save_interval == 0:
                agent.save_model()

    # Start Play for testing
    print("PLAY START")

    # Setup Unity Environment
    engine_configuration_channel = EngineConfigurationChannel()
    env = UnityEnvironment(file_name=env_name,
                           side_channels=[engine_configuration_channel])
    env.reset()

    # setup unity ml agent
    behavior_name = list(env.behavior_specs.keys())[0]
    spec = env.behavior_specs[behavior_name]
    engine_configuration_channel.set_configuration_parameters(time_scale=1.0)
    dec, term = env.get_steps(behavior_name)

    # TEST
    episode, score = 0, 0
    for step in range(test_step):
        state = dec.obs[0]
        action = agent.get_action(state, False)
        action_tuple = ActionTuple()
        action_tuple.add_continuous(action)
        env.set_actions(behavior_name, action_tuple)
        env.step()

        dec, term = env.get_steps(behavior_name)
        done = len(term.agent_id) > 0
        reward = term.reward if done else dec.reward
        next_state = term.obs[0] if done else dec.obs[0]
        score += reward[0]

        if done:
            episode += 1

            # print out
            print(f"{episode} Episode / Step: {step} / Score: {score:.2f} ")
            score = 0

    env.close()



