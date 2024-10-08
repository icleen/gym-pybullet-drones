"""Script demonstrating the use of `gym_pybullet_drones`'s Gymnasium interface.

Classes HoverAviary and MultiHoverAviary are used as learning envs for the PPO algorithm.

Example
-------
In a terminal, run as:

    $ python learn.py --multiagent false
    $ python learn.py --multiagent true

Notes
-----
This is a minimal working example integrating `gym-pybullet-drones` with 
reinforcement learning library `stable-baselines3`.

"""
import os
import time
from datetime import datetime
import argparse
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO, SAC, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from gymnasium.envs.classic_control import Continuous_MountainCarEnv

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.envs.CircleAviary import CircleAviary
from gym_pybullet_drones.envs.SinAviary import SinAviary
from gym_pybullet_drones.envs.TargetAviary import TargetAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('pid') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'
DEFAULT_AGENTS = 1
DEFAULT_MA = False

def run(multiagent=DEFAULT_MA, output_folder=DEFAULT_OUTPUT_FOLDER, gui=DEFAULT_GUI, plot=True, colab=DEFAULT_COLAB, record_video=DEFAULT_RECORD_VIDEO, local=True, model_path=None, rl_alg='ppo', env='circle', no_residual=False, action_steps=1, action_obs=False):

    filename = os.path.join(output_folder, rl_alg + '_save-' + datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    if not os.path.exists(filename):
        os.makedirs(filename+'/')

    target_reward = -0.01
    env_class = CircleAviary
    if env == 'sin':
        env_class = SinAviary
    elif env == 'target':
        env_class = TargetAviary
    elif env == 'mtncar':
        env_class = Continuous_MountainCarEnv
        target_reward = -10
    elif env == 'mtncar':
        env_class = Continuous_MountainCarEnv
        target_reward = -10
    # elif env == 'HalfCheetah-v4':
    #     train_env = make_vec_env(env,
    #                              env_kwargs=dict(),
    #                              n_envs=1,
    #                              seed=0
    #                              )
    #     eval_env = Monitor(gym.make(env, render_mode="rgb_array"))  # Monitor(env_class())

    if env == 'mtncar':
        train_env = make_vec_env(env_class,
                                 env_kwargs=dict(),
                                 n_envs=1,
                                 seed=0
                                 )
        eval_env = Monitor(gym.make(env_class, render_mode="rgb_array"))  # Monitor(env_class())
    else:
        train_env = make_vec_env(env_class,
                                    env_kwargs=dict(
                                        obs=DEFAULT_OBS,
                                        act=DEFAULT_ACT,
                                        action_steps=action_steps,
                                        action_obs=action_obs,
                                        use_residual=not no_residual
                                    ),
                                    n_envs=1,
                                    seed=0
                                    )
        eval_env = Monitor(env_class(
            obs=DEFAULT_OBS,
            act=DEFAULT_ACT,
            action_steps=action_steps,
            action_obs=action_obs,
            use_residual=not no_residual
        ))

    #### Check the environment's spaces ########################
    print(train_env)
    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)
    # train_env.reset()
    # train_env.step(np.zeros(train_env.action_space.shape).reshape(1, 1, -1))
    # print()

    #### Train the model #######################################
    if rl_alg == 'ppo':
        model = PPO('MlpPolicy',
                    train_env,
                    learning_rate=1e-3,
                    # tensorboard_log=filename+'/tb/',
                    verbose=1)
    elif rl_alg == 'sac':
        model = SAC('MlpPolicy',
                    train_env,
                    learning_rate=1e-4,
                    buffer_size=2000000,
                    learning_starts=1000,
                    # tensorboard_log=filename+'/tb/',
                    # action_noise=,
                    verbose=1)
    elif rl_alg == 'a2c':
        model = A2C('MlpPolicy',
                    train_env,
                    # tensorboard_log=filename+'/tb/',
                    verbose=1)
    if model_path is not None:
        model.load(model_path)

    #### Target cumulative rewards (problem-dependent) ##########
    callback_on_best = StopTrainingOnRewardThreshold(
        reward_threshold=target_reward, verbose=1
    )
    eval_callback = EvalCallback(eval_env,
                                 callback_on_new_best=callback_on_best,
                                 verbose=1,
                                 best_model_save_path=filename + '/',
                                 log_path=filename + '/',
                                 eval_freq=int(1000),
                                 deterministic=True,
                                 render=False)
    model.learn(total_timesteps=int(1e7) if local else int(1e2), # shorter training in GitHub Actions pytest
                callback=eval_callback,
                log_interval=100)

    #### Save the model ########################################
    model.save(filename + '/final_model.zip')
    print(filename)

    #### Print training progression ############################
    with np.load(filename + '/evaluations.npz') as data:
        for j in range(data['timesteps'].shape[0]):
            print(str(data['timesteps'][j]) + "," + str(data['results'][j][0]))

    ############################################################
    ############################################################
    ############################################################
    ############################################################
    ############################################################
    return
    if local:
        input("Press Enter to continue...")

    # if os.path.isfile(filename+'/final_model.zip'):
    #     path = filename+'/final_model.zip'
    if os.path.isfile(filename+'/best_model.zip'):
        path = filename+'/best_model.zip'
    else:
        print("[ERROR]: no model under the specified path", filename)
    model = PPO.load(path)

    #### Show (and record a video of) the model's performance ##
    test_env = env_class(
        gui=gui,
        obs=DEFAULT_OBS,
        act=DEFAULT_ACT,
        action_steps=action_steps,
        action_obs=action_obs,
        record=record_video,
        use_residual=True,
    )
    # test_env = env_class(
    #     obs=DEFAULT_OBS,
    #     act=DEFAULT_ACT,
    #     use_residual=True,
    # )
    logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ),
                num_drones=DEFAULT_AGENTS if multiagent else 1,
                output_folder=output_folder,
                colab=colab
                )

    mean_reward, std_reward = evaluate_policy(model,
                                              test_env_nogui,
                                              n_eval_episodes=10
                                              )
    print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")

    obs, info = test_env.reset(seed=42, options={})
    start = time.time()
    for i in range((test_env.EPISODE_LEN_SEC+2)*test_env.CTRL_FREQ):
        action, _states = model.predict(obs,
                                        deterministic=True
                                        )
        obs, reward, terminated, truncated, info = test_env.step(action)
        obs2 = obs.squeeze()
        act2 = action.squeeze()
        print("Obs:", obs, "\tAction", action, "\tReward:", reward, "\tTerminated:", terminated, "\tTruncated:", truncated)
        if DEFAULT_OBS == ObservationType.KIN:
            if not multiagent:
                logger.log(drone=0,
                    timestamp=i/test_env.CTRL_FREQ,
                    state=np.hstack([obs2[0:3],
                                        np.zeros(4),
                                        obs2[3:15],
                                        act2
                                        ]),
                    control=np.zeros(12)
                    )
            else:
                for d in range(DEFAULT_AGENTS):
                    logger.log(drone=d,
                        timestamp=i/test_env.CTRL_FREQ,
                        state=np.hstack([obs2[d][0:3],
                                            np.zeros(4),
                                            obs2[d][3:15],
                                            act2[d]
                                            ]),
                        control=np.zeros(12)
                        )
        test_env.render()
        print(terminated)
        sync(i, start, test_env.CTRL_TIMESTEP)
        if terminated:
            obs = test_env.reset(seed=42, options={})
    test_env.close()

    if plot and DEFAULT_OBS == ObservationType.KIN:
        logger.plot()

if __name__ == '__main__':
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script')
    parser.add_argument('--multiagent',         default=DEFAULT_MA,            type=str2bool,      help='Whether to use example LeaderFollower instead of Hover (default: False)', metavar='')
    parser.add_argument('--gui',                default=DEFAULT_GUI,           type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,  type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB,         type=bool,          help='Whether example is being run by a notebook (default: "False")', metavar='')
    parser.add_argument('--model_path', default=None, type=str, help='path to saved model if starting warm (default: None)', metavar='')
    parser.add_argument('--rl_alg', default='ppo', type=str, help='type of rl algorithm to use (default: "ppo")', metavar='')
    parser.add_argument('--env', default='circle', type=str, help='which environment to train on (default: "circle")', metavar='')
    parser.add_argument('--no_residual', default=False, type=str2bool, help='(default: False)', metavar='')
    parser.add_argument('--action_steps', default=1, type=int, help='(default: 1)', metavar='')
    parser.add_argument('--action_obs', default=False, type=str2bool, help='(default: False)', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
