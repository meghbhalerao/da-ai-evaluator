class MentalModelAugmentation():
    def __init__(self, policy, env, aug_config, num_initial_trajectories=10):
        self.policy = policy
        self.env = env
        self.augmentation_factor = aug_config.factor
        self.num_augmentations = int(self.augmentation_factor * num_initial_trajectories) - num_initial_trajectories

    def augment(self):
        """
        Augment the policy by generating new trajectories using the current policy.
        The number of new trajectories is determined by the augmentation factor.
        """
        augmented_trajectories = []
        for _ in range(self.num_augmentations):
            obs = self.env.reset()
            done = False
            trajectory = []
            while not done:
                action, _ = self.policy.predict(obs)
                next_obs, reward, done, info = self.env.step(action)
                trajectory.append((obs, action, reward, next_obs, done))
                obs = next_obs
            augmented_trajectories.append(trajectory)
        return augmented_trajectories