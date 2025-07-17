import os

# Registering environments:
from gymnasium.envs.registration import register

# Exporting envs:
from flappy_bird_gym.envs.flappy_bird_env_rgb import FlappyBirdEnvRGB

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

register(
    id="FlappyBird-v0",
    entry_point="flappy_bird_gym.envs:FlappyBirdEnvRGB",
)

# Main names:
__all__ = [
    FlappyBirdEnvRGB.__name__,
]