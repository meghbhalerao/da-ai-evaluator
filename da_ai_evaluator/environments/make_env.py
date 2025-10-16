import gymnasium as gym

from flappy_bird_gym.envs.flappy_bird_env_rgb import FlappyBirdEnvRGB
from flappy_bird_gym.envs.flappy_bird_env_simple import FlappyBirdEnvSimple
from gymnasium.envs.registration import register
from gymnasium import make
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv



# Registering environments, in the make env file itself, for better readability and maintainability
register(
    id="FlappyBird-v0",
    entry_point="flappy_bird_gym:FlappyBirdEnvSimple",
)

register(
    id="FlappyBird-rgb-v0",
    entry_point="flappy_bird_gym:FlappyBirdEnvRGB",
)

# Main names:
__all__ = [
    make.__name__,
    FlappyBirdEnvRGB.__name__,
    FlappyBirdEnvSimple.__name__,
]

def make_env(gym_id, env_params=None):
    def _init():
        if gym_id == 'FlappyBird-v0':
            return gym.make(gym_id, **env_params).unwrapped if env_params else gym.make(gym_id).unwrapped
        else:
            raise ValueError(f"Unsupported gym environment ID: {gym_id}")
    return _init
    
def make_vec_env(gym_id, env_params, num_envs, seed, wrapper_class=None):
    if gym_id == 'FlappyBird-v0':
        # train_envs = gym.make_vec(gym_id, num_envs = num_envs, vectorization_mode = 'async', **env_params)
        # eval_envs = gym.make_vec(gym_id, num_envs = num_envs, vectorization_mode = 'async', **env_params)

        train_env = SubprocVecEnv([make_env(gym_id, env_params) for _ in range(num_envs)])
        eval_env = DummyVecEnv([make_env(gym_id, env_params) for _ in range(num_envs)])
        # train_env = gym.vector.AsyncVectorEnv([make_env(gym_id, env_params)] * num_envs)

        # eval_env = gym.vector.AsyncVectorEnv([make_env(gym_id, env_params)] * num_envs)
    else:
        raise ValueError(f"Unsupported gym environment ID: {gym_id}")

    return train_env, eval_env