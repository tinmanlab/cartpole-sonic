"""Small MuJoCo vector environment implementing the original SONIC trainer's env protocol.

Local contributions: CartPole plant, references, observations/reward and reset rules.
There is no local PPO, GAE, rollout-storage or neural-network implementation here.
"""
from pathlib import Path

import mujoco
import numpy as np
from omegaconf import OmegaConf
import torch

STATE_SCALE = np.array([1.8, 3., .55, 4.])
CONTROL_DT = .02


class CartPoleEnv:
    def __init__(self, num_envs=8, seed=9101):
        if not isinstance(num_envs,int) or num_envs<1:
            raise ValueError('num_envs must be positive')
        self.num_envs=num_envs
        self.model=mujoco.MjModel.from_xml_path(str(Path(__file__).with_name('cartpole.xml')))
        self.data=[mujoco.MjData(self.model) for _ in range(num_envs)]
        self.rng=np.random.default_rng(seed)
        self.goals=np.zeros(num_envs)
        self.reference_time=np.zeros(num_envs)
        self.episode_steps=np.zeros(num_envs,dtype=np.int64)
        self.is_evaluating=False
        self.use_symmetry=False
        self.control_transitions=0
        self.physics_steps=0
        self.terminated_episodes=0
        self.last_applied_force=np.zeros(num_envs)
        self.last_terminal_states=np.full((num_envs,4),np.nan)
        self.last_step_states=np.zeros((num_envs,4))
        self.last_targets=np.zeros(num_envs)
        self.config=OmegaConf.create({
            'num_envs':num_envs,
            'robot':{'actions_dim':1,'algo_obs_dim_dict':{'actor_obs':4,'tokenizer':16,'critic_obs':20}},
            'obs':{'obs_dict':{'actor_obs':['state'],'tokenizer':['cartpole_reference'],'critic_obs':['state_and_reference']},
                   'group_obs_dims':{'tokenizer':{'cartpole_reference':[8,2]}},
                   'group_obs_names':{'tokenizer':['cartpole_reference']}},
            'rewards':{'num_critics':1}, 'use_symmetry':False,
        })
        self.reset_all()

    def _reset(self,index):
        d=self.data[index]
        mujoco.mj_resetData(self.model,d)
        d.qpos[:]=self.rng.uniform(-1,1,2)*[.08,.06]
        d.qvel[:]=self.rng.uniform(-.04,.04,2)
        self.goals[index]=self.rng.uniform(-.4,.4)
        self.reference_time[index]=0
        self.episode_steps[index]=0
        mujoco.mj_forward(self.model,d)

    def reset_all(self):
        for i in range(self.num_envs):self._reset(i)
        return self.observations()

    def set_is_evaluating(self,is_evaluating,log_info=False):
        # No domain randomization/symmetry changes in this bounded task.
        self.is_evaluating=bool(is_evaluating)

    def states(self):
        return np.array([[d.qpos[0],d.qvel[0],d.qpos[1],d.qvel[1]] for d in self.data])

    def reference(self,offsets):
        times=self.reference_time[:,None]+np.asarray(offsets)[None,:]
        k=2.5
        decay=np.exp(-k*times)
        x=self.goals[:,None]*(1-(1+k*times)*decay)
        xd=self.goals[:,None]*k*k*times*decay
        return np.stack((x,xd),axis=-1)

    def observations(self):
        state=np.clip(self.states()/STATE_SCALE,-1.5,1.5)
        ref=(self.reference(np.arange(1,9)*.08)/[1.8,3.]).reshape(self.num_envs,16)
        return {'actor_obs':torch.tensor(state,dtype=torch.float32),
                'tokenizer':torch.tensor(ref,dtype=torch.float32),
                'critic_obs':torch.tensor(np.concatenate((state,ref),axis=-1),dtype=torch.float32)}

    @staticmethod
    def reward(states,targets,actions):
        x,xd,theta,thetad=states.T
        return 1-.65*np.minimum((x-targets)**2,4)-8*theta**2-.02*xd**2-.02*thetad**2-.002*actions**2

    def step(self,policy_state_dict):
        raw=policy_state_dict['actions'].detach().cpu().numpy()
        if raw.shape!=(self.num_envs,1) or not np.isfinite(raw).all():
            raise ValueError('Expected finite actions with shape [num_envs,1]')
        action=np.clip(raw[:,0],-1,1) # do not mutate the original Gaussian sample
        self.last_applied_force=action*10.
        for i,d in enumerate(self.data):
            d.ctrl[0]=action[i]
            mujoco.mj_step(self.model,d)
            mujoco.mj_step(self.model,d)
        self.physics_steps+=2*self.num_envs
        self.control_transitions+=self.num_envs
        self.reference_time+=CONTROL_DT
        self.episode_steps+=1
        states=self.states()
        if not np.isfinite(states).all():
            raise RuntimeError('Non-finite MuJoCo state; training is aborted')
        targets=self.reference([0.])[:,0,0]
        rewards=self.reward(states,targets,action)
        dones=(np.abs(states[:,0])>1.78)|(np.abs(states[:,2])>.65)
        self.last_step_states=states.copy()
        self.last_targets=targets.copy()
        self.last_terminal_states=np.where(dones[:,None],states,np.nan)
        self.terminated_episodes+=int(dones.sum())
        metrics={'tracking_mae':torch.tensor(np.abs(states[:,0]-targets),dtype=torch.float32),
                 'abs_pole_angle':torch.tensor(np.abs(states[:,2]),dtype=torch.float32)}
        episode={}
        # The official trainer already computes episode lengths from dones.
        # Do not duplicate it with sparse keys that break process_ep_infos().
        for i in np.flatnonzero(dones):self._reset(int(i))
        # Continuing task: only physical failure terminates. Rollout cutoff is not
        # an environment timeout; the official trainer bootstraps its last value.
        # Time-limit bootstrapping is deliberately not claimed/tested in this gate.
        info={'episode':episode,'time_outs':torch.zeros(self.num_envs,dtype=torch.bool),'to_log':metrics}
        return self.observations(),torch.tensor(rewards,dtype=torch.float32),torch.tensor(dones),info


class CartPoleReconstructionLoss(torch.nn.Module):
    """Task-specific loss binding for the official auxiliary-loss extension point.

    G1 kinematics/alignment losses are not valid for CartPole. Only this tensor
    selection and MSE are local; the original trainer combines/backpropagates it.
    """
    def forward(self, inputs):
        predicted=inputs['decoded_outputs']['cartpole_kin']['cartpole_reference']
        target=inputs['tokenizer_obs']['cartpole_reference']
        return torch.nn.functional.mse_loss(predicted,target)
