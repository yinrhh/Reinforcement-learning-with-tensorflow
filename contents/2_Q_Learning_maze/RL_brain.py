"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import pandas as pd


class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        # 空的Q Table
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    # 策略：智能体根据状态进行下一步动作的函数
    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            # 使用学习的到的策略，也就是Q表
            state_action = self.q_table.loc[observation, :]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            # 10%探索，避免陷入局部最优
            action = np.random.choice(self.actions)
        return action

    # 学习的目的是获得策略函数。用学习出来的策略函数生成概率密度，作为转移概率。Q Learning中的策略函数就是Q Table。
    # 学习更新参数，即更新Q表
    # Q估计 实现Q表的更新
    def learn(self, s, a, r, s_):
        # 检验新的state
        self.check_state_exist(s_)
        # Q估计
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            # 此处选择奖励最大的动作进行判断，但并不是下一步一定要选择的动作
            # Q现实
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )