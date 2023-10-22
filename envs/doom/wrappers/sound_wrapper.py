import gymnasium as gym
import numpy as np
from gymnasium import spaces


class DoomSound(gym.Wrapper):
    """Add game variables to the observation space + reward shaping."""

    def __init__(self, env):
        super().__init__(env)
        current_obs_space = self.observation_space

        self.sampling_rate_int = int(str(self.sampling_freq).split("_")[1])

        # self.unwrapped.skip_frames = 4
        self.aud_len = int((1260 / (44100/self.sampling_rate_int)) * self.unwrapped.number_of_audio_frames)

        audio_shape = [self.aud_len,2]
        sound_high = [[32767,32767]] * self.aud_len
        sound_low = [[-32767,-32767]] * self.aud_len

        if isinstance(self.observation_space, spaces.Dict):
            new_dict = {}
            for space_name, space in self.observation_space.spaces.items():
                new_dict[space_name] = space
            new_dict['sound'] = gym.spaces.Box(
                low=np.array(sound_low, dtype=np.int16), high=np.array(sound_high, dtype=np.int16),
            )
            self.observation_space = gym.spaces.Dict(new_dict) 
        else:
            self.observation_space = gym.spaces.Dict({
                'obs': current_obs_space,
                'sound': gym.spaces.Box(
                    low=np.array(sound_low, dtype=np.int16), high=np.array(sound_high, dtype=np.int16),
                ),
            })

    def reset(self, seed=None, **kwargs):
        base_observation, info = self.env.reset()
        # audio = self.unwrapped.game.get_state().audio_buffer
        audio = self.unwrapped.state.audio_buffer

        if audio is None:
            audio = np.zeros(self.observation_space['sound'].shape)

        elif audio.shape[0] != self.aud_len:
            audio = np.zeros(self.observation_space['sound'].shape)

        if isinstance(base_observation, dict):
            base_observation['sound'] = audio
            return base_observation, info
        else:
            obs_dict = {
                'obs':base_observation,
                # set to zero and run baselines
                'sound':audio
            }
            return obs_dict, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if not terminated:
            audio = self.unwrapped.state.audio_buffer
            # audio = self.unwrapped.game.get_state().audio_buffer
        else:
            audio = np.zeros(self.observation_space['sound'].shape)

        if isinstance(obs, dict):
            obs['sound'] = audio
            return obs, reward, terminated, truncated, info

        else:
            obs_dict = {
                'obs':obs,
                'sound':audio
            }
            return obs_dict, reward, terminated, truncated, info
