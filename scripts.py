import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import random
import time
from matplotlib.colors import ListedColormap


ALPHA = 0.1  # Learning rate
GAMMA = 0.9  # Discount factor
EPSILON = 0.1  # Epsilon-greedy policy
NUM_EPISODES = 400 
MAZE_SIZE = 10
START_STATE = (0, 0)
GOAL_STATE = (9, 9)
REWARD_GOAL = 100
REWARD_STEP = -1
OBSTACLE = -100
ACTIONS = ['up', 'down', 'left', 'right']


class MazeEnv:

    """Maze environment with a 5x5 grid and obstacles."""
    def __init__(self):
        self.grid = np.zeros((MAZE_SIZE, MAZE_SIZE))
        self._add_obstacles()
        self.start_state = START_STATE
        self.goal_state = GOAL_STATE
        self.actions = ACTIONS
        self.state = self.start_state

    def _add_obstacles(self):
        """Random obstacles generation."""
        for i in range(20):
            x, y = random.randint(0,9), random.randint(0,9)
            self.grid[x,y] = OBSTACLE
            
    def reset(self):
        """Reset environment to the start state."""
        self.state = self.start_state
        return self.state

    def step(self, action):
        """Take a step in the environment based on the chosen action."""
        x, y = self.state

        # Move based on the action
        if action == 'up' and x > 0:
            x -= 1
        elif action == 'down' and x < MAZE_SIZE - 1:
            x += 1
        elif action == 'left' and y > 0:
            y -= 1
        elif action == 'right' and y < MAZE_SIZE - 1:
            y += 1

        obstacle_hit = self.grid[x, y] == OBSTACLE
        if obstacle_hit:
            x, y = self.state  # Revert if an obstacle is hit

        self.state = (x, y)
        if self.state == self.goal_state:
            return self.state, REWARD_GOAL, True, obstacle_hit
        return self.state, REWARD_STEP, False, obstacle_hit

Q_TABLE = {
    (x, y): {action: 0 for action in ACTIONS} for x in range(MAZE_SIZE) for y in range(MAZE_SIZE)
}


def q_learning_with_step_visualization(env, num_episodes):
    """Perform Q-learning with step-by-step updates and dynamic visualization."""
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        path = [state]

        print(f"--- Episode {episode + 1} ---")

        while not done:

            if np.random.uniform(0, 1) < EPSILON:
                action = np.random.choice(env.actions)  # Explore
            else:
                action = max(Q_TABLE[state], key=Q_TABLE[state].get)  # Exploit

            next_state, reward, done, obstacle_hit = env.step(action)
            total_reward += reward
            path.append(next_state)

            best_next_action = max(Q_TABLE[next_state], key=Q_TABLE[next_state].get)
            Q_TABLE[state][action] += ALPHA * (
                reward + GAMMA * Q_TABLE[next_state][best_next_action] - Q_TABLE[state][action]
            )

            if episode % 50 == 0  and episode >0 or episode ==9:
                print(
                    f"Step: State {state} -> Action '{action}' -> Next State {next_state} | "
                    f"Reward: {reward}{' (Obstacle hit)' if obstacle_hit else ''}"
                )

                plot_step_by_step(env, path, episode+1, total_reward)

            state = next_state

        if episode % 100 == 0 and episode >0 or episode ==9:
            print(f"Episode {episode + 1} complete - Total Reward: {total_reward}\n")



def plot_step_by_step(env, path, episode, total_reward):
    """Update the maze visualization with obstacles as small squares and the agent's path."""
    maze = np.copy(env.grid)
    plt.ion()
    plt.figure(1, figsize=(8, 8))
    plt.clf() 
    plt.gca().set_facecolor('black')
    obstacles = np.argwhere(maze == OBSTACLE)

    for obs in obstacles:
        x, y = obs
        plt.scatter(y, x, color='#06ff00', s=250, marker='s', zorder=5)


    path_x, path_y = zip(*path)
    plt.plot(path_y, path_x, color='yellow', linewidth=2, label="Agent's Path")
    current_x, current_y = path[-1]
    plt.scatter(current_y, current_x, color='red', s=100, label="Agent Position") 
    plt.scatter(*env.start_state[::-1], marker="p", color="blue", s=100, label="Start")
    plt.scatter(*env.goal_state[::-1], marker="*", color="yellow", s=300, label="Goal")
    plt.title(f"Episode: {episode} | Total Reward: {total_reward}", fontsize=16, color="black")
    plt.legend(loc="upper left", facecolor="white", edgecolor="white", fontsize=10)
    plt.grid(visible=True, color='gray', linestyle='--', linewidth=0.5)
    plt.xticks(range(MAZE_SIZE), color="white")
    plt.yticks(range(MAZE_SIZE), color="white")


    plt.pause(0.0001) 
    plt.draw()

    if episode == NUM_EPISODES: 
        plt.ioff()
        plt.show()



def plot_learning_progress(env, path, q_table, episode):
    """Dynamically update the maze visualization in a single window."""
    maze = np.copy(env.grid)
    
    plt.ion()
    plt.figure(1, figsize=(8, 8))
    plt.clf()  
    
    plt.imshow(maze, cmap='coolwarm', origin='upper', zorder=0)
    
    path_x, path_y = zip(*path)
    plt.plot(path_y, path_x, color='red', linewidth=2, label="Agent's Path")
    plt.scatter(path_y, path_x, color='black', s=50)

    for x in range(MAZE_SIZE):
        for y in range(MAZE_SIZE):
            if (x, y) == env.goal_state or (x, y) in path:
                continue
            action = max(q_table[(x, y)], key=q_table[(x, y)].get)
            dx, dy = (
                (0, -0.3) if action == 'up' else
                (0, 0.3) if action == 'down' else
                (-0.3, 0) if action == 'left' else
                (0.3, 0)
            )
            plt.arrow(y, x, dy, dx, head_width=0.1, head_length=0.1, fc='k', ec='k')

    plt.scatter(*env.start_state[::-1], marker="o", color="green", s=200, label="Start")
    plt.scatter(*env.goal_state[::-1], marker="o", color="yellow", s=300, label="Goal")
    plt.title(f"Episode: {episode}", fontsize=16)
    plt.legend(loc="upper left")
    plt.grid(visible=True, color='gray', linestyle='--', linewidth=0.5)
    plt.xticks(range(MAZE_SIZE))
    plt.yticks(range(MAZE_SIZE))
    
    plt.pause(0.001)  
    plt.draw()

    if episode == NUM_EPISODES:  
        plt.ioff()
        plt.show()



env = MazeEnv()
q_learning_with_step_visualization(env, NUM_EPISODES)
