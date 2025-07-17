import gymnasium as gym

class CollectRollouts:
    def __init__(self, env_id, env_params, policy, num_episodes=10):
        self.env_id = env_id
        self.env = gym.make(self.env_id, **env_params)
        self.policy = policy
        self.num_episodes = num_episodes

    def collect(self):
        rollouts = []
        for _ in range(self.num_episodes):
            obs = self.env.reset()
            done = False
            episode_rollout = []
            while not done:
                action = self.policy.get_action(obs)
                next_obs, reward, done, info = self.env.step(action)
                episode_rollout.append((obs, action, reward, next_obs, done))
                obs = next_obs
            rollouts.append(episode_rollout)
        return rollouts

