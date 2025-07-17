import pygame
import gymnasium as gym

class HumanKeyboardPolicyAgent:
    """
    This is a policy that takes in keyboard inputs to control the Flappy Bird game, and the keys are input by the human player.
    Hence, consequently, the get_action method does not use the observation input, since the human playing the game will be controlling the actions directly, by looking at the screen.
    """
    def __init__(self, cfg):
        pygame.init() # Initialize pygame
        self.env = gym.make(cfg.environment.gym_id, **cfg.environment.env_params)
        self.action_space = self.env.action_space

    def get_action(self, observation): # Note: observation is not used in this policy, since human is controlling the game.
        action = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.KEYDOWN and (
                event.key == pygame.K_SPACE or event.key == pygame.K_UP
            ):
                action = 1
        return action