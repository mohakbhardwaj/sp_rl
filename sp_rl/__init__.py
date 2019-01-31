import os
from gym.envs.registration import register
register(
    id='graphEnvToy-v0',
    entry_point='sp_rl.envs:GraphEnvToy',
    kwargs={'ver' : 0}
)

register(
    id='graphEnvToy-v1',
    entry_point='sp_rl.envs:GraphEnvToy',
    kwargs={'ver' : 1}
)



import register_2d_envs
import register_herb_envs
import register_mixture_envs

