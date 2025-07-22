import gymnasium as gym

def make_env(gym_id, env_params=None):
    if gym_id == 'FlappyBird-v0':
        return gym.make(gym_id, **env_params) if env_params else gym.make(gym_id)
    else:
        raise ValueError(f"Unsupported gym environment ID: {gym_id}")
    
def make_vec_env(gym_id, env_params, num_envs, seed, wrapper_class=None):
    if gym_id == 'FlappyBird-v0':
        train_envs = gym.make_vec(gym_id, num_envs = num_envs, vectorization_mode = 'async', **env_params)
        eval_envs = gym.make_vec(gym_id, num_envs = num_envs, vectorization_mode = 'async', **env_params)
        # train_env = gym.vector.AsyncVectorEnv([make_env(gym_id, env_params)] * num_envs)

        # eval_env = gym.vector.AsyncVectorEnv([make_env(gym_id, env_params)] * num_envs)
    else:
        raise ValueError(f"Unsupported gym environment ID: {gym_id}")

    return train_envs, eval_envs