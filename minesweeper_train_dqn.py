import copy
import os

from lib import models
from lib.wrappers import CustomObservationWrapper
from minesweeper_pygame import *
import argparse
import time
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter

DEFAULT_ENV_NAME = "Minesweeper-v0"
MEAN_REWARD_BOUND = 30.0
WIN_RATE_BOUND = 0.95

# GAMMA = 0.0
# BATCH_SIZE = 64
# REPLAY_SIZE = 100_000
# REPLAY_START_SIZE = 1_000
# LEARNING_RATE = 1e-5
# UPDATE_TARGET_EVERY = 50
# AGG_STATS_EVERY = 100
# SAVE_MODEL_EVERY = 500
# EPSILON_DECAY_LAST_FRAME = 500_000
# EPSILON_START = 1.0
# EPSILON_FINAL = 0.05

# GAMMA = 0.0
# BATCH_SIZE = 128
# REPLAY_SIZE = 200_000
# REPLAY_START_SIZE = 10_000
# LEARNING_RATE = 1e-4
# UPDATE_TARGET_EVERY = 500
# AGG_STATS_EVERY = 100
# SAVE_MODEL_EVERY = 1_000
# EPSILON_DECAY_LAST_FRAME = 500_000
# EPSILON_START = 1.0
# EPSILON_FINAL = 0.01

# GAMMA = 0.0
# BATCH_SIZE = 128
# REPLAY_SIZE = 200_000
# REPLAY_START_SIZE = 10_000
# LEARNING_RATE = 5e-5
# UPDATE_TARGET_EVERY = 200
# AGG_STATS_EVERY = 100
# SAVE_MODEL_EVERY = 1_000
# EPSILON_DECAY_LAST_FRAME = 1000_000
# EPSILON_START = 1.0
# EPSILON_FINAL = 0.01

GAMMA = 0.9
BATCH_SIZE = 128
REPLAY_SIZE = 300_000
REPLAY_START_SIZE = 10_000
LEARNING_RATE = 1e-5
UPDATE_TARGET_EVERY = 100
AGG_STATS_EVERY = 100
SAVE_MODEL_EVERY = 1_000
EPSILON_DECAY_LAST_FRAME = 1_500_000
EPSILON_START = 1.0
EPSILON_FINAL = 0.01

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

model_dir = f"./models/DQN-{int(time.time())}"
os.makedirs(model_dir, exist_ok=True)
log_dir = f"./logs/DQN-{int(time.time())}"
os.makedirs(log_dir, exist_ok=True)


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions, dtype=np.int64), np.array(rewards, dtype=np.float32), \
            np.array(dones, dtype=np.uint8), np.array(next_states)


class Agent:
    def __init__(self, env_dqn, exp_buffer):
        self.env = env_dqn
        self.exp_buffer = exp_buffer
        self.state = env_dqn.reset()
        self.episode_reward = 0.0
        self.episode_count = 0

    def _reset(self):
        self.state = env.reset()
        self.episode_reward = 0.0

    def play_step(self, net_dqn, epsilon_dqn=0.0, device_dqn="cpu"):
        done_reward = None
        done_stats = None

        if np.random.random() < epsilon_dqn:
            # 随机选择任意的单元格(完全随机)
            # action = env.action_space.sample()
            # 随机选择未知的单元格
            board = self.state[-1].reshape(1, env.action_space.n)
            unknown = [i for i, x in enumerate(board[0]) if x == 1]
            action = np.random.choice(unknown)
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device_dqn)
            q_vals_v = net_dqn(state_v)
            # 排除已知的单元格
            q_vals_v = q_vals_v.cpu().detach().numpy()
            mask = self.state[-1].reshape(-1) == 0
            q_vals_v[0][mask] = -float('inf')
            q_vals_v = torch.tensor(q_vals_v)

            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        # do step in the environment
        new_state, step_reward, is_done, _ = self.env.step(action)
        self.episode_reward += step_reward

        exp = Experience(self.state, action, step_reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.episode_reward
            done_stats = copy.deepcopy(self.env.stats)
            self.episode_count += 1
            self._reset()
        return done_reward, done_stats, self.episode_count


def calc_loss(batch_dqn, net_dqn, tgt_net_dqn, device_dqn="cpu"):
    states, actions, rewards, dones, next_states = batch_dqn

    states_v = torch.tensor(states).to(device_dqn)
    next_states_v = torch.tensor(next_states).to(device_dqn)
    actions_v = torch.tensor(actions).to(device_dqn)
    rewards_v = torch.tensor(rewards).to(device_dqn)
    done_mask = torch.BoolTensor(dones).to(device_dqn)

    state_action_values = net_dqn(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_values = tgt_net_dqn(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0
    next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * GAMMA + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)


if __name__ == "__main__":
    # python minesweeper_train_dqn.py --cuda --reward 30.0 --win-rate 0.95
    # tensorboard --logdir=./
    parser = argparse.ArgumentParser(description="Train a DQN to play Minesweeper")
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("--env", default=DEFAULT_ENV_NAME,
                        help="Name of the environment, default=" + DEFAULT_ENV_NAME)
    parser.add_argument("--reward", type=float, default=MEAN_REWARD_BOUND,
                        help="Mean reward boundary for stop of training, default=%.2f" % MEAN_REWARD_BOUND)
    parser.add_argument("--win-rate", type=float, default=WIN_RATE_BOUND,
                        help="Win rate boundary for stop of training, default=%.2f" % WIN_RATE_BOUND)
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    env = MinesweeperEnv(num_rows=6, num_cols=6, num_mines=4, render_mode="rgb_array")
    env = CustomObservationWrapper(env, encoding_type='one_hot')

    net = models.DQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = models.DQN(env.observation_space.shape, env.action_space.n).to(device)
    writer = SummaryWriter(comment="-" + args.env)
    print(net)

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_mean_reward = None
    best_win_rate = None
    list_episode_reward = []
    list_win_rate = []

    while True:
        frame_idx += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)

        reward, stats, n_episode = agent.play_step(net, epsilon, device_dqn=device.type)
        if reward is not None:
            list_episode_reward.append(reward)
            list_win_rate.append(stats['n_win'])

            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()

            # print("%d: done %d games, eps %.2f, speed %.2f f/s" %
            #       (frame_idx, n_episode, epsilon, speed))
            # print(f"episode stats: ", stats)

            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)

            if n_episode % AGG_STATS_EVERY == 0:
                mean_reward = np.mean(list_episode_reward)
                win_rate = np.mean(list_win_rate)
                print("%d: done %d games, mean episode reward %.3f, win rate %.2f" %
                      (frame_idx, n_episode, float(mean_reward), float(win_rate)))
                writer.add_scalar("mean_reward", mean_reward, frame_idx)
                writer.add_scalar("win_rate", win_rate, frame_idx)
                # 如果当前的平均奖励超过了最佳平均奖励,则保存最优奖励模型
                if best_mean_reward is None or best_mean_reward < mean_reward:
                    reward_model_path = os.path.join(model_dir, args.env + "-best_reward.dat")
                    torch.save(net.state_dict(), reward_model_path)
                    if best_mean_reward is not None:
                        print("Best mean reward updated %.3f -> %.3f, model saved"
                              % (float(best_mean_reward), float(mean_reward)))
                    best_mean_reward = mean_reward
                # 如果当前的胜率超过了最佳胜率,则保存最优胜率模型
                if best_win_rate is None or best_win_rate < win_rate:
                    win_rate_model_path = os.path.join(model_dir, args.env + "-best_win_rate.dat")
                    torch.save(net.state_dict(), win_rate_model_path)
                    if best_win_rate is not None:
                        print("Best win rate updated %.3f -> %.3f, model saved"
                              % (float(best_win_rate), float(win_rate)))
                    best_win_rate = win_rate
                # 如果当前的平均奖励和胜率都超过了设定阈值,则问题求解成功
                if mean_reward >= args.reward and win_rate >= args.win_rate:
                    print("Solved in %d frames! Mean reward: %.3f, Win rate: %.3f"
                          % (frame_idx, float(mean_reward), float(win_rate)))
                    break

                list_episode_reward.clear()
                list_win_rate.clear()

            # 定期保存模型
            if n_episode % SAVE_MODEL_EVERY == 0:
                checkpoint_path = f"{args.env}-checkpoint-{n_episode}.dat"
                file_path = os.path.join(model_dir, checkpoint_path)
                torch.save(net.state_dict(), file_path)
                print(f"Checkpoint saved: {checkpoint_path}")

        if len(buffer) < REPLAY_START_SIZE:
            continue

        if frame_idx % UPDATE_TARGET_EVERY == 0:
            tgt_net.load_state_dict(net.state_dict())

        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, net, tgt_net, device_dqn=device.type)
        loss_t.backward()
        optimizer.step()
    writer.close()
    os.system("shutdown")
