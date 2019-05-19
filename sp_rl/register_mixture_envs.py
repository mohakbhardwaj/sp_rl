import os
from gym.envs.registration import register


register(
    id='graphEnvMixture-v1',
    entry_point='sp_rl.envs:GraphEnv2D',
    kwargs={'dataset_folder' : os.path.abspath('../../rss_lsp_datasets/mixture_datasets/dataset_mixture_1'),
            'mode' : 'validation'}
)

register(
    id='graphEnvMixture-v2',
    entry_point='sp_rl.envs:GraphEnv2D',
    kwargs={'dataset_folder' : os.path.abspath('../../rss_lsp_datasets/mixture_datasets/dataset_mixture_2'),
            'mode' : 'validation'}
)

register(
    id='graphEnvMixture-v3',
    entry_point='sp_rl.envs:GraphEnv2D',
    kwargs={'dataset_folder' : os.path.abspath('../../rss_lsp_datasets/mixture_datasets/dataset_mixture_3'),
            'mode' : 'validation'}
)

register(
    id='graphEnvMixture-v4',
    entry_point='sp_rl.envs:GraphEnv2D',
    kwargs={'dataset_folder' : os.path.abspath('../../rss_lsp_datasets/mixture_datasets/dataset_mixture_4'),
            'mode' : 'validation'}
)

register(
    id='graphEnvMixture-v5',
    entry_point='sp_rl.envs:GraphEnv2D',
    kwargs={'dataset_folder' : os.path.abspath('../../rss_lsp_datasets/mixture_datasets/dataset_mixture_5'),
            'mode' : 'validation'}
)

register(
    id='graphEnvMixture-v6',
    entry_point='sp_rl.envs:GraphEnv2D',
    kwargs={'dataset_folder' : os.path.abspath('../../rss_lsp_datasets/mixture_datasets/dataset_mixture_6'),
            'mode' : 'validation'}
)

