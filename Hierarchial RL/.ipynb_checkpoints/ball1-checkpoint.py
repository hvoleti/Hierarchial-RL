import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import pygame

class BallOne(gym.Env):
    def __init__(self):
        super().__init__()
        
        # Define the grid size 3x3
        self.grid_size = 5

        self.max_steps = 50
        self.current_step = 0
        
        # Action space: 0=left, 1=right, 2=up, 3=down
        self.action_space = spaces.Discrete(4)
        
        # Observation space: current position of ball (x,y) in grid (0 to 2)
        self.observation_space = spaces.MultiDiscrete([self.grid_size, self.grid_size])
        
        # Initialize positions
        self.ball_pos = np.array([0, 0])
        self.goal_pos = self._sample_goal()

    def _sample_goal(self):
        # Random position different from ball position
        while True:
            pos = np.array([random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1)])
            if not np.array_equal(pos, self.ball_pos):
                return pos

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
    
        if seed is not None:
            np.random.seed(seed)
    
        self.ball_pos = np.array([0, 0])
        self.goal_pos = self._sample_goal()
        self.current_step = 0
        observation = self.ball_pos
        info = {}  # can include metadata if needed
    
        return observation, info

    
    def step(self, action):
        self.current_step += 1
        # Move the ball according to action within grid bounds
        if action == 0 and self.ball_pos[0] > 0:       # left
            self.ball_pos[0] -= 1
        elif action == 1 and self.ball_pos[0] < self.grid_size - 1:  # right
            self.ball_pos[0] += 1
        elif action == 2 and self.ball_pos[1] > 0:     # up
            self.ball_pos[1] -= 1
        elif action == 3 and self.ball_pos[1] < self.grid_size - 1:  # down
            self.ball_pos[1] += 1
    
        reward = 0
        terminated = False  # True if the task is successfully completed
        truncated = False   # True if the episode ended due to time limit or other constraint
    
        # Check if reached goal
        if np.array_equal(self.ball_pos, self.goal_pos):
            reward = 1
            # The episode does not terminate, just the goal moves
            # You could consider this a success and set terminated=True, depending on your use case
            self.goal_pos = self._sample_goal()

        if self.current_step >= self.max_steps:
            truncated = True
    
        observation = self.ball_pos.copy()
        info = {}
    
        return observation, reward, terminated, truncated, info


    def render(self, mode='human'):
        if not hasattr(self, 'screen'):
            pygame.init()
            self.cell_size = 50
            self.window_size = self.grid_size * self.cell_size
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Ball and Goal Grid")
            self.clock = pygame.time.Clock()
    
        # Colors
        BG_COLOR = (255, 255, 255)
        BALL_COLOR = (0, 0, 255)
        GOAL_COLOR = (0, 255, 0)
        GRID_COLOR = (200, 200, 200)
    
        self.screen.fill(BG_COLOR)
    
        # Draw grid lines
        for x in range(0, self.window_size, self.cell_size):
            pygame.draw.line(self.screen, GRID_COLOR, (x, 0), (x, self.window_size))
        for y in range(0, self.window_size, self.cell_size):
            pygame.draw.line(self.screen, GRID_COLOR, (0, y), (self.window_size, y))
    
        # Draw ball
        ball_rect = pygame.Rect(
            self.ball_pos[0] * self.cell_size,
            self.ball_pos[1] * self.cell_size,
            self.cell_size, self.cell_size
        )
        pygame.draw.ellipse(self.screen, BALL_COLOR, ball_rect)
    
        # Draw goal
        goal_rect = pygame.Rect(
            self.goal_pos[0] * self.cell_size,
            self.goal_pos[1] * self.cell_size,
            self.cell_size, self.cell_size
        )
        pygame.draw.rect(self.screen, GOAL_COLOR, goal_rect)
    
        pygame.display.flip()
        self.clock.tick(10)  # Limit to 10 FPS

    def close(self):
        if hasattr(self, 'screen'):
            pygame.quit()

# # Example usage:
# if __name__ == "__main__":
#     env = BallGoalEnv()
#     obs = env.reset()
#     env.render()
#     actions = [1,1,3,3,0,0,2,2]  # some moves example
#     for a in actions:
#         obs, reward, done, info = env.step(a)
#         env.render()
#         print(f"Reward: {reward}")
