import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from stable_baselines3 import PPO

class BallTwo(gym.Env):
    def __init__(self, ball_one_policy_path="C:\\Users\\hvoleti\\Documents\\Hierarchial RL\\ballone_policy.zip"):
        super().__init__()
        
        # Same grid size as BallOne
        self.grid_size = 5
        self.max_steps = 50
        self.current_step = 0

        # Load BallOne's trained policy
        self.ball_one_policy = PPO.load(ball_one_policy_path)

        self.goal_pos = np.array([0, 0]) 
        
        # BallTwo also moves: left, right, up, down
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiDiscrete([self.grid_size, self.grid_size, 
                                                       self.grid_size, self.grid_size])

        # BallOne and BallTwo positions
        self.ball_one_pos = np.array([0, 0])
        self.ball_two_pos = np.array([self.grid_size - 1, self.grid_size - 1])

    def _sample_goal(self):
        while True:
            pos = np.array([np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size)])
            if not np.array_equal(pos, self.ball_one_pos):
                return pos


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        
        self.ball_one_pos = np.array([0, 0])
        self.ball_two_pos = np.array([self.grid_size - 1, self.grid_size - 1])
        self.current_step = 0
        self.goal_pos = self._sample_goal()

        obs = np.concatenate((self.ball_one_pos, self.ball_two_pos))
        info = {}
        return obs, info

    def step(self, action):
        self.current_step += 1
    
        # Let BallOne act according to its policy
        ball_one_action, _ = self.ball_one_policy.predict(self.ball_one_pos, deterministic=True)
        self._move(self.ball_one_pos, ball_one_action)
    
        # Check if BallOne reached its goal AFTER moving
        if np.array_equal(self.ball_one_pos, self.goal_pos):
            self.goal_pos = self._sample_goal()
    
        # Move BallTwo according to given action
        self._move(self.ball_two_pos, action)
    
        # Reward: negative distance to BallOne (closer is better)
        distance = np.linalg.norm(self.ball_one_pos - self.ball_two_pos, ord=1)
        reward = -distance
    
        terminated = False
        truncated = self.current_step >= self.max_steps
    
        obs = np.concatenate((self.ball_one_pos, self.ball_two_pos))
        info = {"distance": distance}
    
        return obs, reward, terminated, truncated, info


    def _move(self, pos, action):
        if action == 0 and pos[0] > 0: pos[0] -= 1   # left
        elif action == 1 and pos[0] < self.grid_size - 1: pos[0] += 1  # right
        elif action == 2 and pos[1] > 0: pos[1] -= 1   # up
        elif action == 3 and pos[1] < self.grid_size - 1: pos[1] += 1  # down

    def render(self, mode='human'):
        if not hasattr(self, 'screen'):
            pygame.init()
            self.cell_size = 50
            self.window_size = self.grid_size * self.cell_size
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("BallTwo Following BallOne")
            self.clock = pygame.time.Clock()
    
        # Colors
        BG_COLOR = (255, 255, 255)
        BALL_ONE_COLOR = (0, 0, 255)   # Blue
        BALL_TWO_COLOR = (255, 0, 0)   # Red
        GOAL_COLOR = (0, 255, 0)       # Green
        GRID_COLOR = (150, 150, 150)   # Slightly darker grid
    
        self.screen.fill(BG_COLOR)
    
        # Draw grid lines with edges
        for x in range(0, self.window_size + 1, self.cell_size):
            pygame.draw.line(self.screen, GRID_COLOR, (x, 0), (x, self.window_size))
        for y in range(0, self.window_size + 1, self.cell_size):
            pygame.draw.line(self.screen, GRID_COLOR, (0, y), (self.window_size, y))
    
        # Draw goal (green square)
        goal_rect = pygame.Rect(
            self.goal_pos[0] * self.cell_size,
            self.goal_pos[1] * self.cell_size,
            self.cell_size, self.cell_size
        )
        pygame.draw.rect(self.screen, GOAL_COLOR, goal_rect)
    
        # Draw BallOne (blue circle)
        rect1 = pygame.Rect(
            self.ball_one_pos[0] * self.cell_size,
            self.ball_one_pos[1] * self.cell_size,
            self.cell_size, self.cell_size
        )
        pygame.draw.ellipse(self.screen, BALL_ONE_COLOR, rect1)
    
        # Draw BallTwo (red circle)
        rect2 = pygame.Rect(
            self.ball_two_pos[0] * self.cell_size,
            self.ball_two_pos[1] * self.cell_size,
            self.cell_size, self.cell_size
        )
        pygame.draw.ellipse(self.screen, BALL_TWO_COLOR, rect2)
    
        pygame.display.flip()
        self.clock.tick(10)


    def close(self):
        if hasattr(self, 'screen'):
            pygame.quit()
