import flappy_bird_gymnasium
import gymnasium
env = gymnasium.make("FlappyBird-v0", render_mode="rgb_array", use_lidar=False)

obs, _ = env.reset()
while True:
    # Next action:
    # (feed the observation to your agent here)
    action = env.action_space.sample()
    print(env.render().shape)

    # Processing:
    obs, reward, terminated, _, info = env.step(action)

    print(obs.shape)
    
    # Checking if the player is still alive
    if terminated:
        break

env.close()