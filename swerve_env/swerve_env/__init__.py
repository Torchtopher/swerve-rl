from gymnasium.envs.registration import register

register(
    id="SwerveEnv-v0",
    entry_point="swerve_env.envs:SwerveEnv",
)
