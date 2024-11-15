import sys
import time
import torch
import numpy as np
from tqdm import tqdm

from lib import models
from lib.wrappers import CustomObservationWrapper
from minesweeper_pygame import *


# 判断pos是否在扫雷区域board的范围内
def is_in_board(position):
    pos_x, pos_y = position
    board_left = board_origin[0]
    board_top = board_origin[1]
    board_right = board_origin[0] + board_width
    board_bottom = board_origin[1] + board_height
    return board_left <= pos_x <= board_right and board_top <= pos_y <= board_bottom


game_mode = "computer"  # 默认模式为human
if game_mode == "human":
    render_mode = "human"
    env = MinesweeperEnv(HEIGHT, WIDTH, NUM_MINES, render_mode)  # 创建游戏实例
    env = CustomObservationWrapper(env, encoding_type="condensed")
    print(env.board)
    state, reward, done, info = None, None, None, None
    while True:
        action = None
        button = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == Button.LEFT:
                    pos = pygame.mouse.get_pos()
                    if env.reset_button.collidepoint(pos):
                        env.reset()
                        print("reset")
                        print(env.board)
                    if is_in_board(pos) and not env.done:
                        x = (pos[1] - BOARD_PADDING) // brick_size
                        y = (pos[0] - BOARD_PADDING) // brick_size
                        action = x * env.num_cols + y
                        env.button = 'left'
                        print(f"{env.button} button: {x, y}")
                elif event.button == Button.RIGHT and not env.done:
                    pos = pygame.mouse.get_pos()
                    if is_in_board(pos):
                        x = (pos[1] - BOARD_PADDING) // brick_size
                        y = (pos[0] - BOARD_PADDING) // brick_size
                        action = x * env.num_cols + y
                        env.button = 'right'
                        print(f"{env.button} button: {x, y}")
        if action is not None:
            state, reward, done, info = env.step(action)
            print(state, reward, done, info)
        env.render()

if game_mode == "computer":
    render_mode = "rgb_array"
    env = MinesweeperEnv(HEIGHT, WIDTH, NUM_MINES, render_mode)
    env = CustomObservationWrapper(env, encoding_type="one_hot")
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = r"./models/DQN-1726323611/Minesweeper-v0-best_win_rate.dat"  # 替换为你实际的模型路径
    net = models.DQN(env.observation_space.shape, env.action_space.n).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()  # 设置模型为评估模式

    # 设置测试参数
    episodes_per_group = 1000  # 每组1000局
    num_groups = 1000  # 总共1000组
    win_rates = []  # 用于记录每组的胜率

    # 使用进度条显示测试进度
    for group in tqdm(range(1, num_groups + 1), desc="Testing Progress"):
        nWin = 0
        delay = False  # 这里可以设置为 False 以禁用显示
        for episode in range(1, episodes_per_group + 1):
            if render_mode == "human":
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        sys.exit()
            state = env.reset()
            done = False
            episode_rewards = 0  # 记录每个回合的得分
            episode_steps = 0  # 记录每个回合的时间步数
            info = None  # 初始化 info 变量
            first_move = True  # 初始化 first_move 标志
            while not done:
                # 随机选择任意的单元格(完全随机)
                # action = env.action_space.sample()  # this is where you would insert your policy

                # 随机选择未知的单元格, 适用于one_hot和condensed编码
                # board = state[-1].reshape(1, env.action_space.n)
                # unknown = [i for i, x in enumerate(board[0]) if x == 1]
                # action = np.random.choice(unknown)

                if first_move:
                    action = 0  # 第一次移动时点击左上角
                    first_move = False  # 更新 first_move 标志
                # elif np.random.random() < 0.01:
                #     board = state[-1].reshape(1, env.action_space.n)
                #     unknown = [i for i, x in enumerate(board[0]) if x == 1]
                #     action = np.random.choice(unknown)
                else:
                    # 使用模型预测动作
                    with torch.no_grad():
                        input_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                        output_tensor = net(input_tensor)
                        probability_array = output_tensor.cpu().detach().numpy()
                        probability_array = np.squeeze(probability_array)

                        mask = state[-1].reshape(-1) == 0
                        probability_array_flat = probability_array.reshape(-1)
                        probability_array_flat[mask] = -float('inf')
                        action = np.argmax(probability_array_flat)

                env.button = 'left'
                next_state, reward, done, info = env.step(action)
                state = next_state
                episode_rewards += reward
                episode_steps += 1
                env.render()
                if delay:
                    time.sleep(0.5)
            if delay:
                time.sleep(2)
            if info["status"] == "win":
                nWin += 1
            # print(f"Episode:{episode} Rewards:{episode_rewards:.2f} Steps:{episode_steps} Total Wins:{nWin} ")
            # print(f"episode stats: ", env.stats)
        # 计算并记录当前组的胜率
        win_rate = nWin / episodes_per_group
        win_rates.append(win_rate)
    # 统计平均胜率和标准差
    average_win_rate = np.mean(win_rates)
    std_dev_win_rate = np.std(win_rates)

    print(f"Average Win Rate over {num_groups} groups: {average_win_rate:.4f}")
    print(f"Standard Deviation of Win Rate: {std_dev_win_rate:.4f}")
    env.close()
