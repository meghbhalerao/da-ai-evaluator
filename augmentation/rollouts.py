class CollectRollouts:
    def __init__(self, env, policy, num_episodes=10):
        self.env = env
        self.policy = policy
        self.num_episodes = num_episodes

    def collect(self):
        rollouts = []
        for _ in range(self.num_episodes):
            obs = self.env.reset()
            done = False
            episode_rollout = []
            while not done:
                action = self.policy(obs)
                next_obs, reward, done, info = self.env.step(action)
                episode_rollout.append((obs, action, reward, next_obs, done))
                obs = next_obs
            rollouts.append(episode_rollout)
        return rollouts

