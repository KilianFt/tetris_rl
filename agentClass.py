from turtle import position
import numpy as np
import random
import math
import h5py
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# This file provides the skeleton structure for the classes TQAgent and TDQNAgent to be completed by you, the student.
# Locations starting with # TO BE COMPLETED BY STUDENT indicates missing code that should be written by you.

class TQAgent:
    # Agent for learning to play tetris using Q-learning
    def __init__(self,alpha,epsilon,episode_count):
        # Initialize training parameters
        self.alpha=alpha
        self.epsilon=epsilon
        self.episode=0
        self.episode_count=episode_count

    def fn_init(self,gameboard):
        self.gameboard=gameboard
        self.q_tables = {}
        self.initial_q_table_shape = {}

        self.reward_tots = np.zeros(self.episode_count)

        # get all the possible tiles
        for i, tile in enumerate(gameboard.tiles):
            n_orientations = len(tile)
            actions = {}

            for or_idx in range(n_orientations):
                n_positions = 1 + gameboard.N_col - len(tile[or_idx])
                actions[or_idx] = np.zeros(n_positions)

            self.initial_q_table_shape[i] = actions
            self.q_tables[i] = {}

        self.cur_board_str = ''
        self.action = (-1, -1)
        self.tile_idx = -1


    def fn_load_strategy(self,strategy_file):
        pass
        # TO BE COMPLETED BY STUDENT
        # Here you can load the Q-table (to Q-table of self) from the input parameter strategy_file (used to test how the agent plays)

    def get_board_str(self):
        board_str = ''
        for x in self.gameboard.board.flatten():
            board_str += str(x)

        return board_str

    def fn_read_state(self):
        self.board_str = self.get_board_str()
        self.tile_idx = self.gameboard.cur_tile_type

    def get_q_table(self, board_str, tile_idx):
        if board_str in self.q_tables[tile_idx]:
            q_table = self.q_tables[tile_idx][board_str]
        else:
            # initialize new 0 q table
            q_table = self.initial_q_table_shape[tile_idx]
            self.q_tables[tile_idx][board_str] = copy.deepcopy(q_table)
        return q_table

    def get_max_q(self, q_table):
        q_max = None
        
        for positions in q_table.values():
            curr_max = np.max(positions)
            if q_max is None:
                q_max = curr_max
            if curr_max > q_max:
                q_max = curr_max

        return q_max

    def get_best_actions(self, q_table):
        max_q = self.get_max_q(q_table)
        max_idxs = []
        for o_idx, positions in q_table.items():
            for p_idx, q_val in enumerate(positions):
                if q_val == max_q:
                    max_idxs.append((o_idx, p_idx))

        return max_idxs

    def fn_select_action(self):
        # Useful variables: 
        # 'self.epsilon' parameter epsilon in epsilon-greedy policy

        self.old_state = self.board_str

        # get q table for current state
        q_table = self.get_q_table(self.board_str, self.tile_idx)

        max_idxs = self.get_best_actions(q_table)
        n_max = len(max_idxs)
        rand_idx = np.random.randint(0, n_max)

        r = np.random.rand()
        if r < self.epsilon:
            # chose random action
            o_len = len(q_table)
            rand_o = np.random.randint(0, o_len)

            p_len = len(q_table[rand_o])
            rand_p = np.random.randint(0, p_len)

            self.action = (rand_o, rand_p)
        else:
            self.action = max_idxs[rand_idx]

        if self.gameboard.fn_move(self.action[1], self.action[0]) == 1:
            print("move invalid")

    def fn_reinforce(self,old_state,reward):
        last_board_str = old_state[0]
        last_tile_idx = old_state[1]

        q_table_next_state = self.get_q_table(self.board_str, last_tile_idx)
        max_next_state = self.get_max_q(q_table_next_state)

        # get entry in q table of current state with current action
        q_table_state = self.get_q_table(last_board_str, last_tile_idx)

        q_state = q_table_state[self.action[0]][self.action[1]]

        q_state_updated = q_state + self.alpha * (reward + max_next_state - q_state)

        self.q_tables[last_tile_idx][last_board_str][self.action[0]][self.action[1]] = copy.deepcopy(q_state_updated)

        return


    def fn_turn(self):
        if self.gameboard.gameover:
            self.episode+=1
            if self.episode%100==0:
                print('episode '+str(self.episode)+'/'+str(self.episode_count)+' (reward: ',str(np.sum(self.reward_tots[range(self.episode-100,self.episode)])),')')
            if self.episode%1000==0:
                saveEpisodes=[1000,2000,5000,10000,20000,50000,100000,200000,500000,1000000];
                if self.episode in saveEpisodes:
                    np.save('rewards.npy', self.reward_tots)
                    np.save('q_tables.npy', self.q_tables)
                    
            if self.episode>=self.episode_count:
                raise SystemExit(0)
            else:
                self.gameboard.fn_restart()
        else:
            # Select and execute action (move the tile to the desired column and orientation)
            self.fn_select_action()

            old_state = [copy.deepcopy(self.board_str), copy.deepcopy(self.tile_idx)]

            # Drop the tile on the game board
            reward=self.gameboard.fn_drop()
            self.reward_tots[self.episode] += reward

            # Read the new state
            self.fn_read_state()
            # Update the Q-table using the old state and the reward (the new state and the taken action should be stored as attributes in self)
            self.fn_reinforce(old_state, reward)


class Net(nn.Module):
    def __init__(self, state_size, action_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

def criterion(q_val, y):
    loss = torch.square((q_val - y))
    return loss

class TDQNAgent:
    # Agent for learning to play tetris using Q-learning
    def __init__(self,alpha,epsilon,epsilon_scale,replay_buffer_size,batch_size,sync_target_episode_count,episode_count):
        # Initialize training parameters
        self.alpha=alpha
        self.epsilon=epsilon
        self.epsilon_scale=epsilon_scale
        self.replay_buffer_size=replay_buffer_size
        self.batch_size=batch_size
        self.sync_target_episode_count=sync_target_episode_count
        self.episode=0
        self.episode_count=episode_count

    def fn_init(self,gameboard):
        self.gameboard=gameboard
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # In this function you could set up and initialize the states, actions, the Q-networks (one for calculating actions and one target network), experience replay buffer and storage for the rewards
        # You can use any framework for constructing the networks, for example pytorch or tensorflow
        # This function should not return a value, store Q network etc as attributes of self

        # Useful variables: 
        # 'gameboard.N_row' number of rows in gameboard
        # 'gameboard.N_col' number of columns in gameboard
        # 'len(gameboard.tiles)' number of different tiles
        # 'self.alpha' the learning rate for stochastic gradient descent
        # 'self.episode_count' the total number of episodes in the training
        # 'self.replay_buffer_size' the number of quadruplets stored in the experience replay buffer

        state_size = gameboard.N_row * gameboard.N_col
        # indexes are pos.or, so 2.3 would be position 2 with orientation 3
        # [1.1, 1.2, 1.3, 1.4, 2.1 ...]
        action_size = gameboard.N_col * 4

        self.q_nn = Net(state_size = state_size, action_size = action_size)

        self.q_nn_hat = copy.deepcopy(self.q_nn)

        learning_rate = 1e-3
        self.optimizer = optim.Adam(self.q_nn.parameters(), lr=learning_rate)

        possible_actions = {}
        for i, tile in enumerate(gameboard.tiles):
            tile_actions = {}
            for o_idx in range(len(tile)):
                n_positions = 1 + gameboard.N_col - len(tile[o_idx])
                tile_actions[o_idx] = n_positions

            possible_actions[i] = tile_actions

        self.possible_actions = possible_actions

    def fn_load_strategy(self, strategy_file):
        pass
        # TO BE COMPLETED BY STUDENT
        # Here you can load the Q-network (to Q-network of self) from the strategy_file

    def fn_read_state(self):
        
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # In this function you could calculate the current state of the gane board
        # You can for example represent the state as a copy of the game board and the identifier of the current tile
        # This function should not return a value, store the state as an attribute of self

        self.board = self.gameboard.board.flatten()
        self.tile_idx = self.gameboard.cur_tile_type


    def fn_select_action(self):
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # Choose and execute an action, based on the output of the Q-network for the current state, or random if epsilon greedy
        # This function should not return a value, store the action as an attribute of self and exectute the action by moving the tile to the desired position and orientation

        # Useful variables: 
        # 'self.epsilon' parameter epsilon in epsilon-greedy policy
        # 'self.epsilon_scale' parameter for the scale of the episode number where epsilon_N changes from unity to epsilon

        curr_e_scale = 1 - self.episode / self.epsilon_scale
        curr_epsilon = np.max([self.epsilon, curr_e_scale])

        tile_actions = self.possible_actions[self.tile_idx]

        r = np.random.rand()
        if r < curr_epsilon:
            # chose random action
            o_rand = np.random.randint(0, len(tile_actions))
            p_rand = np.random.randint(0, len(tile_actions[o_rand]))
            action = (p_rand, o_rand)
            self.gameboard.fn_move(p_rand, o_rand)

        else:
            # chose action with highest q_val
            one_hot_tile = [i if i == self.tile else 0 for i in range(len(self.gameboard.tiles)) ]
            flat_state = np.hstack([self.board, one_hot_tile])
            output = self.q_nn(flat_state)

            max_idxs = np.argsort(output)
            for idx in max_idxs:
                o_idx = idx % 4
                p_idx = np.floor(idx / 4)
                if not self.gameboard.fn_move(p_idx, o_idx) == 1:
                    action = (p_idx, o_idx)
                    break


    def fn_reinforce(self,batch):
        pass
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # Update the Q network using a batch of quadruplets (old state, last action, last reward, new state)
        # Calculate the loss function by first, for each old state, use the Q-network to calculate the values Q(s_old,a), i.e. the estimate of the future reward for all actions a
        # Then repeat for the target network to calculate the value \hat Q(s_new,a) of the new state (use \hat Q=0 if the new state is terminal)
        # This function should not return a value, the Q table is stored as an attribute of self

        # Useful variables: 
        # The input argument 'batch' contains a sample of quadruplets used to update the Q-network

        # in your training loop:
        self.optimizer.zero_grad()   # zero the gradient buffers
        output = self.q_nn(input)
        if epison_ended:
            target = reward
        else:
            max_q_nn_hat
            target = reward + max_q_nn_hat
        loss = criterion(output, target)
        loss.backward()
        self.optimizer.step()    # Does the update

    def fn_turn(self):
        if self.gameboard.gameover:
            self.episode+=1
            if self.episode%100==0:
                print('episode '+str(self.episode)+'/'+str(self.episode_count)+' (reward: ',str(np.sum(self.reward_tots[range(self.episode-100,self.episode)])),')')
            if self.episode%1000==0:
                saveEpisodes=[1000,2000,5000,10000,20000,50000,100000,200000,500000,1000000];
                if self.episode in saveEpisodes:
                    pass
                    # TO BE COMPLETED BY STUDENT
                    # Here you can save the rewards and the Q-network to data files
            if self.episode>=self.episode_count:
                raise SystemExit(0)
            else:
                if (len(self.exp_buffer) >= self.replay_buffer_size) and ((self.episode % self.sync_target_episode_count)==0):
                    pass
                    # TO BE COMPLETED BY STUDENT
                    # Here you should write line(s) to copy the current network to the target network
                    self.q_nn_hat = copy.deepcopy(self.q_nn)

                self.gameboard.fn_restart()
        else:
            # Select and execute action (move the tile to the desired column and orientation)
            self.fn_select_action()
            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to copy the old state into the variable 'old_state' which is later stored in the ecperience replay buffer

            # Drop the tile on the game board
            reward=self.gameboard.fn_drop()

            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to add the current reward to the total reward for the current episode, so you can save it to disk later

            # Read the new state
            self.fn_read_state()

            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to store the state in the experience replay buffer

            if len(self.exp_buffer) >= self.replay_buffer_size:
                # TO BE COMPLETED BY STUDENT
                # Here you should write line(s) to create a variable 'batch' containing 'self.batch_size' quadruplets 
                self.fn_reinforce(batch)


class THumanAgent:
    def fn_init(self,gameboard):
        self.episode=0
        self.reward_tots=[0]
        self.gameboard=gameboard

    def fn_read_state(self):
        pass

    def fn_turn(self,pygame):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit(0)
            if event.type==pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.reward_tots=[0]
                    self.gameboard.fn_restart()
                if not self.gameboard.gameover:
                    if event.key == pygame.K_UP:
                        self.gameboard.fn_move(self.gameboard.tile_x,(self.gameboard.tile_orientation+1)%len(self.gameboard.tiles[self.gameboard.cur_tile_type]))
                    if event.key == pygame.K_LEFT:
                        self.gameboard.fn_move(self.gameboard.tile_x-1,self.gameboard.tile_orientation)
                    if event.key == pygame.K_RIGHT:
                        self.gameboard.fn_move(self.gameboard.tile_x+1,self.gameboard.tile_orientation)
                    if (event.key == pygame.K_DOWN) or (event.key == pygame.K_SPACE):
                        self.reward_tots[self.episode]+=self.gameboard.fn_drop()