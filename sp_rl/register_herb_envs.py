import os
from gym.envs.registration import register


register(
    id='graphEnvHerb-v1',
    entry_point='sp_rl.envs:GraphEnvHerb',
    kwargs={'dataset_folder' : os.path.abspath('../../rss_lsp_datasets/dataset_herb_1'),
            'mode' : 'train'}
)

register(
    id='graphEnvHerbValidation-v1',
    entry_point='sp_rl.envs:GraphEnvHerb',
    kwargs={'dataset_folder' : os.path.abspath('../../rss_lsp_datasets/dataset_herb_1'),
            'mode' : 'validation'}
)

register(
    id='graphEnvHerb-v2',
    entry_point='sp_rl.envs:GraphEnvHerb',
    kwargs={'dataset_folder' : os.path.abspath('../../rss_lsp_datasets/dataset_herb_2'),
            'mode' : 'train'}
)

register(
    id='graphEnvHerbValidation-v2',
    entry_point='sp_rl.envs:GraphEnvHerb',
    kwargs={'dataset_folder' : os.path.abspath('../../rss_lsp_datasets/dataset_herb_2'),
            'mode' : 'validation'}
)


register(
    id='graphEnvHerb-v3',
    entry_point='sp_rl.envs:GraphEnvHerb',
    kwargs={'dataset_folder' : os.path.abspath('../../rss_lsp_datasets/dataset_herb_3'),
            'mode' : 'train'}
)

register(
    id='graphEnvHerbValidation-v3',
    entry_point='sp_rl.envs:GraphEnvHerb',
    kwargs={'dataset_folder' : os.path.abspath('../../rss_lsp_datasets/dataset_herb_3'),
            'mode' : 'validation'}
)


register(
    id='graphEnvHerb-v4',
    entry_point='sp_rl.envs:GraphEnvHerb',
    kwargs={'dataset_folder' : os.path.abspath('../../rss_lsp_datasets/dataset_herb_4'),
            'mode' : 'train'}
)

register(
    id='graphEnvHerbValidation-v4',
    entry_point='sp_rl.envs:GraphEnvHerb',
    kwargs={'dataset_folder' : os.path.abspath('../../rss_lsp_datasets/dataset_herb_4'),
            'mode' : 'validation'}
)

register(
    id='graphEnvHerb-v5',
    entry_point='sp_rl.envs:GraphEnvHerb',
    kwargs={'dataset_folder' : os.path.abspath('../../rss_lsp_datasets/dataset_herb_5'),
            'mode' : 'train'}
)

register(
    id='graphEnvHerbValidation-v5',
    entry_point='sp_rl.envs:GraphEnvHerb',
    kwargs={'dataset_folder' : os.path.abspath('../../rss_lsp_datasets/dataset_herb_5'),
            'mode' : 'validation'}
)