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

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

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
        self.possible_tile_actions = {}

        self.reward_tots = np.zeros(self.episode_count)

        # get all the possible tiles
        for i, tile in enumerate(gameboard.tiles):
            n_orientations = len(tile)
            actions = {}

            for or_idx in range(n_orientations):
                n_positions = 1 + gameboard.N_col - len(tile[or_idx])
                actions[or_idx] = n_positions

            self.possible_tile_actions[i] = actions

            self.q_tables[i] = {}

        self.cur_board_str = ''
        self.action = (-1, -1)
        self.tile_idx = -1


    def fn_load_strategy(self,strategy_file):
        self.q_tables = np.load(strategy_file)

    def get_board_str(self):
        board_str = ''
        for x in self.gameboard.board.flatten():
            if x == 1.0:
                board_str += str(1)
            else:
                board_str += str(0)

        return board_str

    def fn_read_state(self):
        self.board_str = self.get_board_str()
        self.tile_idx = self.gameboard.cur_tile_type

    def get_possible_actions(self, tile_idx):
        actions = []
        poses = self.possible_tile_actions[tile_idx]
        for o_idx, n_positions in poses.items():
            for p_idx in range(n_positions):
                actions.append((p_idx, o_idx))

        return actions

    def get_action_rewards(self, actions, board_str):
        rewards = {}
        q_table = self.q_tables[self.tile_idx]

        for action in actions:
            rewards[action] = 0.0
            if board_str in q_table:
                if action in q_table[board_str]:
                    rewards[action] = q_table[board_str][action]

        return rewards

    def fn_select_action(self):

        q_table = self.q_tables[self.tile_idx]
        possible_actions = self.get_possible_actions(self.tile_idx)

        if not self.board_str in q_table:
            self.action = random.choice(possible_actions)

        else:
            rewards = self.get_action_rewards(possible_actions, self.board_str)
            best_actions = []
            max_reward = max(rewards.values())
            for action, reward in rewards.items():
                if reward >= max_reward:
                    best_actions.append(action)

            if np.random.rand() < self.epsilon:
                self.action = random.choice(possible_actions)
            else:
                self.action = random.choice(best_actions)

        if self.gameboard.fn_move(self.action[0], self.action[1]) == 1:
            print("move invalid")


    def fn_reinforce(self,old_state,reward):
        last_board_str = old_state[0]
        last_tile_idx = old_state[1]

        new_board_str = self.board_str
        new_tile_idx = self.tile_idx

        action = self.action

        q_table_new_tile = self.q_tables[new_tile_idx]
        q_table_last_tile = self.q_tables[last_tile_idx]

        max_q_new_state = 0
        if new_board_str in q_table_new_tile:
            possible_actions = self.get_possible_actions(self.tile_idx)
            rewards = self.get_action_rewards(possible_actions, self.board_str)
            max_reward = max(rewards.values())
            max_q_new_state = max_reward


        last_q_val = 0
        if last_board_str in q_table_last_tile:
            if action in q_table_last_tile[last_board_str]:
                last_q_val = q_table_last_tile[last_board_str][action]

        q_state_updated = last_q_val + self.alpha * (reward + max_q_new_state - last_q_val)

        if last_board_str in q_table_last_tile:
            q_table_last_tile[last_board_str][action] = q_state_updated
        else:
            q_table_last_tile[last_board_str] = {action: q_state_updated}

    def fn_turn(self):
        if self.gameboard.gameover:
            self.episode+=1
            if self.episode%100==0:
                print('episode '+str(self.episode)+'/'+str(self.episode_count)+' (reward: ',str(np.sum(self.reward_tots[range(self.episode-100,self.episode)])),')')
            if self.episode%1000==0:
                saveEpisodes=[1000,2000,5000,10000,20000,50000,100000,200000,500000,1000000];
                if self.episode in saveEpisodes:
                    np.save(("data/test_1c_"+str(self.episode_count)+"_step_"+str(self.episode)+'_rewards.npy'), self.reward_tots)
                    np.save(("data/test_1c_"+str(self.episode_count)+"_step_"+str(self.episode)+'_q_tables.npy'), self.q_tables)
                    
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
    def __init__(self, state_size, action_size, hidden_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc3(x)
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
        n_tiles = len(gameboard.tiles)
        state_size = gameboard.N_row * gameboard.N_col + n_tiles
        # indexes are pos.or, so 2.3 would be position 2 with orientation 3
        # [1.1, 1.2, 1.3, 1.4, 2.1 ...]
        action_size = gameboard.N_col * 4

        self.action_size = action_size
        self.state_size = state_size
        print("state size", state_size)
        hidden_size = 128

        self.q_nn = Net(state_size = state_size, action_size = action_size, hidden_size = hidden_size)
        print(self.q_nn)

        self.q_nn_hat = copy.deepcopy(self.q_nn)

        learning_rate = 1e-4
        self.optimizer = optim.Adam(self.q_nn.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        possible_actions = {}
        for i, tile in enumerate(gameboard.tiles):
            tile_actions = {}
            for o_idx in range(len(tile)):
                n_positions = 1 + gameboard.N_col - len(tile[o_idx])
                tile_actions[o_idx] = n_positions

            possible_actions[i] = tile_actions

        self.possible_actions = possible_actions

        self.exp_buffer = []
        self.reward_tots = np.zeros(self.episode_count)

    def fn_load_strategy(self, strategy_file):
        self.q_nn = torch.load(strategy_file)
        self.q_nn_hat = copy.deepcopy(self.q_nn)

    def fn_read_state(self):
        self.board = self.gameboard.board.flatten()
        self.tile_idx = self.gameboard.cur_tile_type

        one_hot_tile = [1 if i == self.tile_idx else 0 for i in range(len(self.gameboard.tiles))]
        flat_state = np.hstack([self.board, one_hot_tile])
        self.state = torch.from_numpy(flat_state).to(torch.float32)

    def fn_select_action(self):
        curr_e_scale = 1 - self.episode / self.epsilon_scale
        curr_epsilon = np.max([self.epsilon, curr_e_scale])

        tile_actions = self.possible_actions[self.tile_idx]

        r = np.random.rand()
        if r < curr_epsilon:
            # chose random action
            o_rand = np.random.randint(0, len(tile_actions))
            p_rand = np.random.randint(0, tile_actions[o_rand])
            self.action = (p_rand, o_rand)
            self.gameboard.fn_move(p_rand, o_rand)

        else:
            # chose action with highest q_val
            with torch.no_grad():
                output = self.q_nn(self.state)

            max_idxs = np.flip(np.argsort(output.detach().numpy()))
            for idx in max_idxs:
                o_idx = idx % self.gameboard.N_col
                p_idx = idx // self.gameboard.N_col
                if not self.gameboard.fn_move(p_idx, o_idx) == 1:
                    self.action = (p_idx, o_idx)
                    break


    def fn_reinforce(self, batch):
        # in your training loop:
        actions = []
        rewards = []
        old_states = torch.Tensor(self.batch_size, self.state_size)
        new_states = torch.Tensor(self.batch_size, self.state_size)
        for i, entry in enumerate(batch):
            rewards.append(entry['reward'])
            old_state = entry['old_state']
            new_state = entry['new_state']
            actions.append(entry['action'])
            old_states[i] = old_state
            new_states[i] = new_state

        self.optimizer.zero_grad()

        output = self.q_nn(old_states)
        output_hat = self.q_nn_hat(new_states)
        
        target_vals = torch.Tensor(self.batch_size, 1)
        output_vals = torch.Tensor(self.batch_size, 1)
        # targets = torch.Tensor(self.batch_size, self.action_size)

        targets = torch.clone(output)

        for i, (action, reward) in enumerate(zip(actions, rewards)):
            reward_pos = action[0] * self.gameboard.N_col + action[1]
            # find out if state is terminal state
            out_hat = output_hat[i]
            
            # FIXME should also consider case of 50 rounds
            if reward == -100:
                target_reward = reward

            else:
                
                max_q_nn_hat = torch.max(out_hat)
                target_reward = reward + max_q_nn_hat

            target_vals[i] = target_reward
            output_vals[i] = output[i][reward_pos]

            targets[i][reward_pos] = target_reward

        # target_vals.requires_grad = True
        # output_vals.requires_grad = True

        # loss = criterion(output, targets)
        loss = self.criterion(output, targets)
        # loss.mean().backward()
        loss.backward()

        self.optimizer.step()

    def fn_turn(self):
        if self.gameboard.gameover:
            self.episode += 1
            if self.episode % 100==0:
                print('episode '+str(self.episode)+'/'+str(self.episode_count)+' (reward: ',str(np.sum(self.reward_tots[range(self.episode-100,self.episode)])),')')
            if self.episode % 1000 == 0:
                saveEpisodes=[1000,2000,5000,10000,20000,50000,100000,200000,500000,1000000];
                if self.episode in saveEpisodes:
                    np.save(("data/test_2a_"+str(self.episode_count)+"_step_"+str(self.episode)+'_rewards.npy'), self.reward_tots)
                    model_save_dir = "data/test_2a_"+str(self.episode_count)+"_step_"+str(self.episode)+"_qnn.pt"
                    torch.save(self.q_nn, model_save_dir)

            if self.episode>=self.episode_count:
                raise SystemExit(0)
            else:
                if (len(self.exp_buffer) >= self.replay_buffer_size) and ((self.episode % self.sync_target_episode_count)==0):
                    self.q_nn_hat = copy.deepcopy(self.q_nn)

                self.gameboard.fn_restart()
        else:
            # Select and execute action (move the tile to the desired column and orientation)
            self.fn_select_action()

            old_state = copy.deepcopy(self.state)

            # Drop the tile on the game board
            reward = self.gameboard.fn_drop()

            self.reward_tots[self.episode] += reward

            # Read the new state
            self.fn_read_state()

            new_state = self.state
            action = self.action
            quadruplet = {'old_state': old_state,
                'action': action,
                'reward': reward,
                'new_state': new_state}

            if len(self.exp_buffer) < self.replay_buffer_size:
                self.exp_buffer.append(copy.deepcopy(quadruplet))
            else:
                self.exp_buffer.pop(0)
                self.exp_buffer.append(copy.deepcopy(quadruplet))

                print("ERROR wrong buffer size: ", len(self.exp_buffer)) if not len(self.exp_buffer) == self.replay_buffer_size else 0

            if len(self.exp_buffer) >= self.replay_buffer_size:
                batch = random.choices(self.exp_buffer, k = self.batch_size)
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