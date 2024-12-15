import numpy as np
import gym, torch
import os
from gym.wrappers import RecordVideo

# Thiết lập EGL cho MuJoCo (tránh lỗi OpenGL)
os.environ["MUJOCO_GL"] = "egl"

cpu_num = 1
os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)
import time 
import random
import psutil

import argparse
import pickle
from EvoRainbow_core.parameters import Parameters

parser = argparse.ArgumentParser()
parser.add_argument('-env', required=True, type=str,
                    help='Environment Choices: (Swimmer-v2) (HalfCheetah-v2) (Hopper-v2) (Walker2d-v2) (Ant-v2)')
parser.add_argument('-seed', type=int, default=7)
parser.add_argument('-pr', type=int, default=128)
parser.add_argument('-pop_size', type=int, default=10)
parser.add_argument('-disable_cuda', action='store_true')
parser.add_argument('-render', action='store_true', help='Render gym episodes (record video)')
parser.add_argument('-sync_period', type=int)
parser.add_argument('-novelty', action='store_true')
parser.add_argument('-proximal_mut', action='store_true')
parser.add_argument('-distil', action='store_true')
parser.add_argument('-distil_type', type=str, default='fitness')
parser.add_argument('-EA', action='store_true')
parser.add_argument('-RL', action='store_true')
parser.add_argument('-detach_z', action='store_true')
parser.add_argument('-random_choose', action='store_true')
parser.add_argument('-per', action='store_true')
parser.add_argument('-use_all', action='store_true')
parser.add_argument('-intention', action='store_true')
parser.add_argument('-mut_mag', type=float, default=0.05)
parser.add_argument('-tau', type=float, default=0.005)
parser.add_argument('-prob_reset_and_sup', type=float, default=0.05)
parser.add_argument('-frac', type=float, default=0.1)
parser.add_argument('-TD3_noise', type=float, default=0.2)
parser.add_argument('-mut_noise', action='store_true')
parser.add_argument('-verbose_mut', action='store_true')
parser.add_argument('-verbose_crossover', action='store_true')
parser.add_argument('-logdir', type=str, required=True)
parser.add_argument('-opstat', action='store_true')
parser.add_argument('-opstat_freq', type=int, default=1)
parser.add_argument('-save_periodic', action='store_true')
parser.add_argument('-next_save', type=int, default=200)
parser.add_argument('-K', type=int, default=5)
parser.add_argument('-OFF_TYPE', type=int, default=1)
parser.add_argument('-num_evals', type=int, default=1)
parser.add_argument('-version', type=int, default=1)
parser.add_argument('-time_steps', type=int, default=1)
parser.add_argument('-test_operators', action='store_true')
parser.add_argument('-EA_actor_alpha', type=float, default=1.0)
parser.add_argument('-state_alpha', type=float, default=1.0)
parser.add_argument('-actor_alpha', type=float, default=1.0)
parser.add_argument('-theta', type=float, default=0.5)
parser.add_argument('-gamma', type=float, default=0.99)
parser.add_argument('-scale', type=float, default=1.0)
parser.add_argument('-EA_tau', type=float, default=0.3)
parser.add_argument('-Soft_Update', default=1, type=int)
parser.add_argument('-Value_Function', default=1, type=int)
parser.add_argument('-n_grad', default=3, type=int)
parser.add_argument('-sigma_init', default=1e-3, type=float)
parser.add_argument('-damp', default=1e-3, type=float)
parser.add_argument('-damp_limit', default=1e-5, type=float)
parser.add_argument('-elitism', action='store_true')
parser.add_argument('-mult_noise', action='store_true')

# Thêm chế độ direct_load
parser.add_argument('-direct_load', action='store_true',
                    help='Nếu bật cờ này, bỏ qua huấn luyện, load mô hình từ .pkl và tạo video luôn')
parser.add_argument('-actor_pkl_path', type=str,
                    help='Đường dẫn đến file pkl chứa trọng số actor nếu dùng direct_load')
parser.add_argument('-state_emb_pkl_path', type=str,
                    help='Đường dẫn đến file pkl chứa trọng số state embedding nếu dùng direct_load')
parser.add_argument('-episodes_for_video', type=int, default=3,
                    help='Số episode để ghi video trong chế độ direct_load hoặc sau huấn luyện')

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
parameters = Parameters(parser)  # Inject the cli arguments in the parameters object

# Tạo môi trường với render_mode="rgb_array"
env = gym.make(parameters.env_name, render_mode="rgb_array")

if parameters.render:
    video_folder = "./videos"
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)
    # RecordVideo với render_mode="rgb_array"
    env = RecordVideo(env, video_folder=video_folder, 
                      episode_trigger=lambda episode_id: True)
    print(f"Lưu video quá trình chơi vào: {video_folder}")

print("env.action_space.low", env.action_space.low, "env.action_space.high", env.action_space.high)
parameters.action_dim = env.action_space.shape[0]
parameters.state_dim = env.observation_space.shape[0]

# Write the parameters to the info file and print them
parameters.write_params(stdout=True)

# Seed
os.environ['PYTHONHASHSEED']= str(parameters.seed)
env.reset(seed=parameters.seed)
torch.manual_seed(parameters.seed)
np.random.seed(parameters.seed)
random.seed(parameters.seed)

from EvoRainbow_core import mod_utils as utils, agent

env.action_space.seed(parameters.seed)

if __name__ == "__main__":
    if parameters.direct_load:
        # Chế độ direct_load: Bỏ qua huấn luyện
        env.close()
        print("Chế độ direct_load: Bỏ qua huấn luyện, load mô hình từ .pkl và tạo video luôn.")

        if parameters.actor_pkl_path is None or parameters.state_emb_pkl_path is None:
            raise ValueError("Phải truyền -actor_pkl_path và -state_emb_pkl_path khi dùng -direct_load")

        # Tạo môi trường video mới
        test_env = gym.make(parameters.env_name, render_mode="rgb_array")
        test_video_folder = "./videos_direct_load"
        if not os.path.exists(test_video_folder):
            os.makedirs(test_video_folder)
        test_env = RecordVideo(test_env, video_folder=test_video_folder,
                               episode_trigger=lambda episode_id: True)
        test_env.reset(seed=parameters.seed)

        # Khởi tạo agent để biết kiến trúc mô hình
        temp_agent = agent.Agent(parameters, test_env)
        test_actor = type(temp_agent.muagent.actor)(parameters)  # Tạo instance actor mới
        test_state_emb = type(temp_agent.rl_agent.state_embedding)(parameters)  # Tạo instance state_emb mới

        test_actor.load_state_dict(torch.load(parameters.actor_pkl_path, map_location='cpu'))
        test_state_emb.load_state_dict(torch.load(parameters.state_emb_pkl_path, map_location='cpu'))

        test_actor.eval()
        test_state_emb.eval()

        # Chạy vài episode để tạo video
        for ep in range(parameters.episodes_for_video):
            obs = test_env.reset()
            done = False
            ep_reward = 0
            while not done:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                obs_emb = test_state_emb(obs_tensor)
                with torch.no_grad():
                    # Giả sử actor.forward(obs_emb) trả về action trực tiếp
                    action = test_actor(obs_emb).cpu().numpy().squeeze()
                action = np.clip(action, test_env.action_space.low, test_env.action_space.high)
                obs, reward, done, info = test_env.step(action)
                ep_reward += reward
            print(f"Episode {ep+1}, Reward: {ep_reward}")

        test_env.close()
        print(f"Video chơi mô hình từ direct_load đã được lưu tại {test_video_folder}")

    else:
        # Chế độ huấn luyện
        agent_instance = agent.Agent(parameters, env)
        print('Running', parameters.env_name, ' State_dim:', parameters.state_dim, ' Action_dim:', parameters.action_dim)

        next_save = parameters.next_save
        time_start = time.time()
        while agent_instance.num_frames <= parameters.num_frames:
            stats = agent_instance.train()
            # ... (Phần huấn luyện, logging và lưu mô hình như code gốc) ...
            
            if agent_instance.num_games > next_save:
                next_save += parameters.next_save
                actor_pkl_path = os.path.join(parameters.save_foldername, str(parameters.seed) + '_mu_actor_'+str(agent_instance.num_games)+'.pkl')
                state_emb_pkl_path = os.path.join(parameters.save_foldername, str(parameters.seed) + '_evo_share_'+str(agent_instance.num_games)+'.pkl')
                torch.save(agent_instance.muagent.actor.state_dict(), actor_pkl_path)
                torch.save(agent_instance.rl_agent.state_embedding.state_dict(), state_emb_pkl_path)

                torch.save(agent_instance.rl_agent.actor.state_dict(), os.path.join(parameters.save_foldername,
                                                               str(parameters.seed) + '_evo_rl_actor_'+str(agent_instance.num_games)+'.pkl'))
                
                if parameters.save_periodic:
                    save_folder = os.path.join(parameters.save_foldername, 'models')
                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)

                    buffer_save_name = os.path.join(save_folder, 'champion_buffer_{}.pkl'.format(next_save))
                    with open(buffer_save_name, 'wb+') as buffer_file:
                        pickle.dump(agent_instance.rl_agent.buffer, buffer_file)

                print("Progress Saved")

        # Huấn luyện kết thúc, đóng môi trường huấn luyện
        env.close()
        print("Huấn luyện kết thúc, video huấn luyện (nếu có) đã được finalize.")

        # Tạo video sau huấn luyện
        test_env = gym.make(parameters.env_name, render_mode="rgb_array")
        test_video_folder = "./videos_after_training"
        if not os.path.exists(test_video_folder):
            os.makedirs(test_video_folder)
        test_env = RecordVideo(test_env, video_folder=test_video_folder,
                               episode_trigger=lambda episode_id: True)
        test_env.reset(seed=parameters.seed)

        # Load mô hình từ file pkl cuối cùng đã lưu
        # Giả định actor_pkl_path, state_emb_pkl_path là file cuối cùng đã lưu
        test_actor = type(agent_instance.muagent.actor)(parameters)
        test_state_emb = type(agent_instance.rl_agent.state_embedding)(parameters)

        test_actor.load_state_dict(torch.load(actor_pkl_path, map_location='cpu'))
        test_state_emb.load_state_dict(torch.load(state_emb_pkl_path, map_location='cpu'))

        test_actor.eval()
        test_state_emb.eval()

        for ep in range(parameters.episodes_for_video):
            obs = test_env.reset()
            done = False
            ep_reward = 0
            while not done:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                obs_emb = test_state_emb(obs_tensor)
                with torch.no_grad():
                    action = test_actor(obs_emb).cpu().numpy().squeeze()
                action = np.clip(action, test_env.action_space.low, test_env.action_space.high)
                obs, reward, done, info = test_env.step(action)
                ep_reward += reward
            print(f"Episode {ep+1}, Reward: {ep_reward}")

        test_env.close()
        print(f"Video chơi mô hình sau huấn luyện đã được lưu tại {test_video_folder}")
