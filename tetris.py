import numpy as np
import pygame
import h5py
import gameboardClass
import agentClass

PARAM_TASK1a=1
PARAM_TASK1b=2
PARAM_TASK1c=3
PARAM_TASK1d=4
PARAM_TASK2a=5
PARAM_TASK2b=6

# Choose to control the game yourself ('human_player=1') to test the setups in the different tasks
human_player=0
human_player=1

# Choose parameter sets for different tasks
param_set=PARAM_TASK1a
#param_set=PARAM_TASK1b
#param_set=PARAM_TASK1c
#param_set=PARAM_TASK1d
#param_set=PARAM_TASK2a
#param_set=PARAM_TASK2b

# Use files to evaluate strategy
# If you change 'strategy_file' to the location of a file containing a stored Q-table or Q-network, you can evaluate the success of the found strategy
if param_set==PARAM_TASK1a:
    strategy_file=''
elif param_set==PARAM_TASK1b:
    strategy_file=''
elif param_set==PARAM_TASK1c:
    strategy_file=''
elif param_set==PARAM_TASK2a:
    strategy_file=''
elif param_set==PARAM_TASK2b:
    strategy_file=''

if strategy_file:
    evaluate_agent=1
    human_player=1
else:
    evaluate_agent=0


# The code below initializes the game parameters for the task selected by 'param_set'
# Game parameters: 
# 'N_row' and 'N_col' (integers) gives the size of the game board.
# 'tile_size' (2 or 4) denotes whether the small tile set (2) or the large tile set (4) should be used
# 'max_tile_count' (integer) denotes the maximal number of tiles to be placed in one game
# 'stochastic_prob' (float between 0 and 1) denotes the probability to take a random tile. When stochastic_prob=0 tiles are taken according to a predefined sequence, when stochastic_prob=1 all tiles are random. For values 0<stochastic_prob<1 there is a mixture between deterministic and random tiles

# Training parameters:
# 'alpha' is learning rate in Q-learning or for the stochastic gradient descent in deep Q-networks
# 'epsilon' is probability to choose random action in epsilon-greedy policy
# 'episode_count' is the number of epsiodes a training session lasts

# Additional training parameters for deep Q-networks:
# 'epsilon_scale' is the scale of the episode number where epsilon_N changes from unity to epsilon
# 'replay_buffer_size' is the size of the experience replay buffer
# 'batch_size' is the number of samples taken from the experience replay buffer each update
# 'sync_target_episode_count' is the number of epsiodes between synchronisations of the target network
if param_set==PARAM_TASK1a:
    N_row=4
    N_col=4
    tile_size=2
    max_tile_count=50
    stochastic_prob=0

    alpha=0.2
    epsilon=0
    episode_count=1000

    if (not human_player) or evaluate_agent:
        agent=agentClass.TQAgent(alpha,epsilon,episode_count)
elif param_set==PARAM_TASK1b:
    N_row=4
    N_col=4
    tile_size=2
    max_tile_count=50
    stochastic_prob=0

    alpha=0.2
    epsilon=0.001
    episode_count=10000

    if (not human_player) or evaluate_agent:
        agent=agentClass.TQAgent(alpha,epsilon,episode_count)
elif param_set==PARAM_TASK1c:
    N_row=4
    N_col=4
    tile_size=2
    max_tile_count=50
    stochastic_prob=1

    alpha=0.2
    epsilon=0.001
    episode_count=200000

    if (not human_player) or evaluate_agent:
        agent=agentClass.TQAgent(alpha,epsilon,episode_count)
elif param_set==PARAM_TASK1d:
    N_row=8
    N_col=8
    tile_size=4
    max_tile_count=50
    stochastic_prob=1

    alpha=0.2
    epsilon=0.001
    episode_count=200000

    if (not human_player) or evaluate_agent:
        agent=agentClass.TQAgent(alpha,epsilon,episode_count)
elif param_set==PARAM_TASK2a:
    N_row=4
    N_col=4
    tile_size=2
    max_tile_count=50
    stochastic_prob=1

    alpha=0.001
    epsilon=0.001
    episode_count=10000

    epsilon_scale=5000
    replay_buffer_size=10000
    batch_size=32
    sync_target_episode_count=100

    if (not human_player) or evaluate_agent:
        agent=agentClass.TDQNAgent(alpha,epsilon,epsilon_scale,replay_buffer_size,batch_size,sync_target_episode_count,episode_count)
elif param_set==PARAM_TASK2b:
    N_row=8
    N_col=8
    tile_size=4
    max_tile_count=50
    stochastic_prob=1

    alpha=0.001
    epsilon=0.001
    episode_count=10000

    epsilon_scale=50000

    replay_buffer_size=10000
    batch_size=32
    sync_target_episode_count=100

    if (not human_player) or evaluate_agent:
        agent=agentClass.TDQNAgent(alpha,epsilon,epsilon_scale,replay_buffer_size,batch_size,sync_target_episode_count,episode_count)
else:
    print('Erroneouse param_set. Terminating...')
    raise SystemExit(0)

# The remaining code below is implementation of the game. You don't need to change anything below this line

if evaluate_agent:
    agent_evaluate=agent;
if human_player:
    agent=agentClass.THumanAgent()
        
gameboard=gameboardClass.TGameBoard(N_row,N_col,tile_size,max_tile_count,agent,stochastic_prob)

if evaluate_agent:
    agent_evaluate.epsilon=0
    agent_evaluate.fn_init(gameboard)
    agent_evaluate.fn_load_strategy(strategy_file)

if isinstance(gameboard.agent,agentClass.THumanAgent):
    # The player is human

    # Define some colors for painting
    COLOR_BLACK = (0, 0, 0)
    COLOR_GREY = (128, 128, 128)
    COLOR_WHITE = (255, 255, 255)
    COLOR_RED =  (255, 0, 0)

    # Initialize the game engine
    pygame.init()
    screen=pygame.display.set_mode((200+N_col*20,150+N_row*20))
    clock=pygame.time.Clock()
    pygame.key.set_repeat(300,100)
    pygame.display.set_caption('Turn-based tetris')
    font=pygame.font.SysFont('Calibri',25,True)
    fontLarge=pygame.font.SysFont('Calibri',50,True)
    framerate=0;

    # Loop until the window is closed
    while True:
        if isinstance(gameboard.agent,agentClass.THumanAgent):
            gameboard.agent.fn_turn(pygame)
        else:
            pygame.event.pump()
            for event in pygame.event.get():
                if event.type==pygame.KEYDOWN:
                    if event.key==pygame.K_SPACE:
                        if framerate > 0:
                            framerate=0
                        else:
                            framerate=10
                    if (event.key==pygame.K_LEFT) and (framerate>1):
                        framerate-=1
                    if event.key==pygame.K_RIGHT:
                        framerate+=1
            gameboard.agent.fn_turn()

        if evaluate_agent:
            agent_evaluate.fn_read_state()
            agent_evaluate.fn_select_action()

        if pygame.display.get_active():
            # Paint game board
            screen.fill(COLOR_WHITE)

            for i in range(gameboard.N_row):
                for j in range(gameboard.N_col):
                    pygame.draw.rect(screen,COLOR_GREY,[100+20*j,80+20*(gameboard.N_row-i),20,20],1)
                    if gameboard.board[i][j] > 0:
                        pygame.draw.rect(screen,COLOR_BLACK,[101+20*j,81+20*(gameboard.N_row-i),18,18])

            if gameboard.cur_tile_type is not None:
                curTile=gameboard.tiles[gameboard.cur_tile_type][gameboard.tile_orientation]
                for xLoop in range(len(curTile)):
                    for yLoop in range(curTile[xLoop][0],curTile[xLoop][1]):
                        pygame.draw.rect(screen,COLOR_RED,[101+20*((xLoop+gameboard.tile_x)%gameboard.N_col),81+20*(gameboard.N_row-(yLoop+gameboard.tile_y)),18,18])

            screen.blit(font.render("Reward: "+str(agent.reward_tots[agent.episode]),True,COLOR_BLACK),[0,0])
            screen.blit(font.render("Tile "+str(gameboard.tile_count)+"/"+str(gameboard.max_tile_count),True,COLOR_BLACK),[0,20])
            if framerate>0:
                screen.blit(font.render("FPS: "+str(framerate),True,COLOR_BLACK),[320,0])
            screen.blit(font.render("Reward: "+str(agent.reward_tots[agent.episode]),True,COLOR_BLACK),[0,0])
            if gameboard.gameover:
                screen.blit(fontLarge.render("Game Over", True,COLOR_RED), [80, 200])
                screen.blit(font.render("Press ESC to try again", True,COLOR_RED), [85, 265])

            pygame.display.flip()
            clock.tick(framerate)
else:
    # The player is AI
    while True:
        gameboard.agent.fn_turn()




