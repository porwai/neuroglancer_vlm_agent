from ngllib import Environment

# Initialize options for environmental interaction, including state return types
options = {
        'euler_angles': True,
        'resize': False,
        'add_mouse': False,
        'fast': True,
        'image_path': None
}

def custom_reward(state, action, prev_state):
    return 1, False

env = Environment(headless=False, config_path="config.json", verbose=False, reward_function=custom_reward)

env.start_session(**options)

for i in range(100):
    
    # action_vector should reflect a model output; here it is hardcoded for demonstration purposes
    action_vector = [
        0, 0, 0,  # left, right, double click booleans
        100, 100,  # x, y
        0, 0, 0,  # no modifier keys
        1,  # no JSON change
        10, 0, 0,  # position change
        0,  # cross-section scaling
        0.2, 0, 0,  # orientation change in Euler angles, which is better for a model to learn or a human to understand
        2000  # projection scaling (log-scale in neuroglancer)
        ]
    
    _, reward, _, _ = env.step(action_vector)

    print(reward)