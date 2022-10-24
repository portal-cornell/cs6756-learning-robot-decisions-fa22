import argparse
import numpy as np
from tqdm import trange
import gym
from gym_simplifiedtetris.envs import SimplifiedTetrisBinaryEnv as Tetris

def test_policy(obs, env):
    """ Policy to be evaluated.

    For more details on action and observation space, refer to:
    https://github.com/OliverOverend/gym-simplifiedtetris/tree/dev/gym_simplifiedtetris/envs

    Args:
        obs: The observation representing the tetris board as a 1D array
         + the ID of the piece.
        For instance, to view the tetris board as you see it rendered
            print(obs[:-1].reshape(10,20).T)
        env: 

    Return:
        The action to execute
    """
    #TODO: Please call your trained policy here.
    # For now, we randomly sample an action which isn't all that smart
    return env.action_space.sample()


def heuristic_policy(obs, env):
    # Load heuristic agent (Pierre Dellacherie's algorithm)
    from gym_simplifiedtetris.agents import DellacherieAgent
    agent = DellacherieAgent()
    return agent.predict(env)

def evaluate_policy(num_episodes: int, max_steps: int, policy_type: str, render: bool) -> None:
    """ Evaluates learned tetris policy

    Args:
        num_episodes: Number of evaluation episodes
        max_steps: Maximum number of steps after which game is terminated
        policy: Which policy to evaluate
        render: If True, render evaluate
    """
    # Create environment
    env = Tetris(grid_dims=(20, 10), piece_size=4)
    obs = env.reset()
    # Policy to evaluate
    policy = None
    if policy_type == 'test':
        policy = test_policy
    elif policy_type == 'heuristic':
        policy = heuristic_policy
    # Log total score in each episode
    ep_scores = np.zeros(num_episodes)

    for episode in range(0, num_episodes):
        obs = env.reset()
        score = 0
        for step in trange(0, max_steps):
            if render: env.render()
            action = policy(obs, env)
            obs, reward, done, info = env.step(action)
            score += reward
            if done:
                break

        ep_scores[episode] = score
        print(f"Episode: {episode}  Total score: {ep_scores[episode]}")
    env.close()

    print(f"Summary over {num_episodes} epsiodes:  Highest score: {np.max(ep_scores)} Mean score: {np.mean(ep_scores)} Std score: {np.std(ep_scores)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test trained Tetris policy.')
    parser.add_argument('--num_episodes', type=int, default=10, help='Number of evaluation episodes')
    parser.add_argument('--max_steps', type=int, default=10000, help='Maximum number of steps after which game terminates')
    parser.add_argument('--policy', type=str, choices=['test', 'heuristic'], default='test', help='Type of policy to evaluate')
    parser.add_argument('--render', action='store_true', default=False, help='Render environment')

    args = parser.parse_args()
    evaluate_policy(num_episodes=args.num_episodes, max_steps=args.max_steps, policy_type=args.policy, render=args.render)
