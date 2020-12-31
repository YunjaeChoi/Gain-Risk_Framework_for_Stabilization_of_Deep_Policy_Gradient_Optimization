import numpy as np
from utils.tf_utils import *
from utils.math_utils import *

class AdvantageBuffer:

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.max_size = size
        self.obs_buf_shape = combined_shape(size, obs_dim)
        self.act_buf_shape = combined_shape(size, act_dim)

        self.obs_buf = np.zeros(self.obs_buf_shape, dtype=np.float32)
        self.act_buf = np.zeros(self.act_buf_shape, dtype=np.float32)
        self.act_dn_buf = np.zeros(self.act_buf_shape, dtype=np.float32)
        self.next_obs_buf = np.zeros(self.obs_buf_shape, dtype=np.float32)
        self.logp_buf = np.zeros(self.max_size, dtype=np.float32)

        self.act_mu_dn_buf = np.zeros(self.act_buf_shape, dtype=np.float32)
        self.act_std_dn_buf = np.zeros(self.act_buf_shape, dtype=np.float32)

        self.rew_buf = np.zeros(self.max_size, dtype=np.float32)
        self.done_buf = np.zeros(self.max_size, dtype=np.float32)
        self.adv_buf = np.zeros(self.max_size, dtype=np.float32)
        self.ret_buf = np.zeros(self.max_size, dtype=np.float32)

        self.next_ret_buf = np.zeros(self.max_size, dtype=np.float32)
        self.cut_off_ep_len = 0

        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx = 0, 0
        self.path_slices = []

    def reset_buffer(self):
        self.ptr, self.path_start_idx = 0, 0
        self.path_slices = []
        self.cut_off_ep_len = 0

    def store(self, obs, act, act_dn, act_mu_dn, act_std_dn, next_obs, logp, rew, done):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.act_dn_buf[self.ptr] = act_dn
        self.next_obs_buf[self.ptr] = next_obs
        self.logp_buf[self.ptr] = logp

        self.act_mu_dn_buf[self.ptr] = act_mu_dn
        self.act_std_dn_buf[self.ptr] = act_std_dn

        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr += 1

    def store_cut_off_ep_len(self, cut_off_ep_len):
        self.cut_off_ep_len = cut_off_ep_len

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        self.path_slices.append(path_slice)
        rews = np.append(self.rew_buf[path_slice], last_val)
        rets = discount_cumsum(rews, self.gamma)
        self.ret_buf[path_slice] = rets[:-1]
        self.next_ret_buf[path_slice] = rets[1:]
        self.path_start_idx = self.ptr

    def get_vf_update_inputs(self):
        assert self.ptr == self.max_size
        input_idx_length = self.max_size - self.cut_off_ep_len
        obs_slice = self.obs_buf[:input_idx_length]
        ret_slice = self.ret_buf[:input_idx_length]

        done_idx = self.done_buf.astype(np.bool)
        done_obs = self.next_obs_buf[done_idx]
        done_ret = self.next_ret_buf[done_idx]

        obs_concat = np.concatenate([obs_slice, done_obs])
        ret_concat = np.concatenate([ret_slice, done_ret])
        vf_inputs = [
            obs_concat,
            ret_concat
        ]
        return vf_inputs

    def get_observations(self):
        assert self.ptr == self.max_size
        return [self.obs_buf, self.next_obs_buf]

    def store_vf_outputs(self, obs_v, next_obs_v):
        obs_v = obs_v.reshape(-1)
        next_obs_v = next_obs_v.reshape(-1)
        assert len(next_obs_v) == self.max_size
        pds = self.gamma * next_obs_v - obs_v
        deltas = self.rew_buf + pds

        for i, ps in enumerate(self.path_slices):
            #GAE lambda
            self.adv_buf[ps] = discount_cumsum(deltas[ps], self.gamma * self.lam)

            #advantage normalization
            if len(self.adv_buf[ps]) > 1:
                adv_mean = np.mean(self.adv_buf[ps])
                adv_std = np.std(self.adv_buf[ps])
                self.adv_buf[ps] = (self.adv_buf[ps] - adv_mean) / (adv_std + 1e-8)
            else:
                self.adv_buf[ps] = 0.0

    def get_pi_update_inputs(self):
        assert self.ptr == self.max_size
        self.reset_buffer()
        return [self.obs_buf, self.act_dn_buf, self.act_mu_dn_buf,
                self.act_std_dn_buf, self.logp_buf, self.adv_buf]
