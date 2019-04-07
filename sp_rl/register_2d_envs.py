import os
from gym.envs.registration import register


register(
    id='graphEnv2D-v1',
    entry_point='sp_rl.envs:GraphEnv2D',
    kwargs={'dataset_folder' : os.path.abspath('../../graph_collision_checking_dataset/dataset_2d_1'),
            'mode' : 'train'}
)

register(
    id='graphEnv2DValidation-v1',
    entry_point='sp_rl.envs:GraphEnv2D',
    kwargs={'dataset_folder' : os.path.abspath('../../graph_collision_checking_dataset/dataset_2d_1'),
            'mode' : 'validation'}

)

register(
    id='graphEnv2DTest-v1',
    entry_point='sp_rl.envs:GraphEnv2D',
    kwargs={'dataset_folder' : os.path.abspath('../../graph_collision_checking_dataset/dataset_2d_1'),
            'mode' : 'test'}

)

register(
    id='graphEnv2D-v2',
    entry_point='sp_rl.envs:GraphEnv2D',
    kwargs={'dataset_folder' : os.path.abspath('../../graph_collision_checking_dataset/dataset_2d_2'),
            'mode' : 'train'}
)

register(
    id='graphEnv2DValidation-v2',
    entry_point='sp_rl.envs:GraphEnv2D',
    kwargs={'dataset_folder' : os.path.abspath('../../graph_collision_checking_dataset/dataset_2d_2'),
            'mode' : 'validation'}

)

register(
    id='graphEnv2DTest-v2',
    entry_point='sp_rl.envs:GraphEnv2D',
    kwargs={'dataset_folder' : os.path.abspath('../../graph_collision_checking_dataset/dataset_2d_2'),
            'mode' : 'test'}

)


register(
    id='graphEnv2D-v3',
    entry_point='sp_rl.envs:GraphEnv2D',
    kwargs={'dataset_folder' : os.path.abspath('../../graph_collision_checking_dataset/dataset_2d_3'),
            'mode' : 'train'}
)

register(
    id='graphEnv2DValidation-v3',
    entry_point='sp_rl.envs:GraphEnv2D',
    kwargs={'dataset_folder' : os.path.abspath('../../graph_collision_checking_dataset/dataset_2d_3'),
            'mode' : 'validation'}

)

register(
    id='graphEnv2DTest-v3',
    entry_point='sp_rl.envs:GraphEnv2D',
    kwargs={'dataset_folder' : os.path.abspath('../../graph_collision_checking_dataset/dataset_2d_3'),
            'mode' : 'test'}

)


register(
    id='graphEnv2D-v4',
    entry_point='sp_rl.envs:GraphEnv2D',
    kwargs={'dataset_folder' : os.path.abspath('../../graph_collision_checking_dataset/dataset_2d_4'),
            'mode' : 'train'}
)

register(
    id='graphEnv2DValidation-v4',
    entry_point='sp_rl.envs:GraphEnv2D',
    kwargs={'dataset_folder' : os.path.abspath('../../graph_collision_checking_dataset/dataset_2d_4'),
            'mode' : 'validation'}
)

register(
    id='graphEnv2DTest-v4',
    entry_point='sp_rl.envs:GraphEnv2D',
    kwargs={'dataset_folder' : os.path.abspath('../../graph_collision_checking_dataset/dataset_2d_4'),
            'mode' : 'test'}
)



register(
    id='graphEnv2D-v5',
    entry_point='sp_rl.envs:GraphEnv2D',
    kwargs={'dataset_folder' : os.path.abspath('../../graph_collision_checking_dataset/dataset_2d_5'),
            'mode' : 'train'}
)

register(
    id='graphEnv2DValidation-v5',
    entry_point='sp_rl.envs:GraphEnv2D',
    kwargs={'dataset_folder' : os.path.abspath('../../graph_collision_checking_dataset/dataset_2d_5'),
            'mode' : 'validation'}
)

register(
    id='graphEnv2DTest-v5',
    entry_point='sp_rl.envs:GraphEnv2D',
    kwargs={'dataset_folder' : os.path.abspath('../../graph_collision_checking_dataset/dataset_2d_5'),
            'mode' : 'test'}
)

register(
    id='graphEnv2D-v6',
    entry_point='sp_rl.envs:GraphEnv2D',
    kwargs={'dataset_folder' : os.path.abspath('../../graph_collision_checking_dataset/dataset_2d_6'),
            'mode' : 'train'}
)

register(
    id='graphEnv2DValidation-v6',
    entry_point='sp_rl.envs:GraphEnv2D',
    kwargs={'dataset_folder' : os.path.abspath('../../graph_collision_checking_dataset/dataset_2d_6'),
            'mode' : 'validation'}
)

register(
    id='graphEnv2DTest-v6',
    entry_point='sp_rl.envs:GraphEnv2D',
    kwargs={'dataset_folder' : os.path.abspath('../../graph_collision_checking_dataset/dataset_2d_6'),
            'mode' : 'test'}
)


register(
    id='graphEnv2D-v7',
    entry_point='sp_rl.envs:GraphEnv2D',
    kwargs={'dataset_folder' : os.path.abspath('../../graph_collision_checking_dataset/dataset_2d_7'),
            'mode' : 'train'}
)

register(
    id='graphEnv2DValidation-v7',
    entry_point='sp_rl.envs:GraphEnv2D',
    kwargs={'dataset_folder' : os.path.abspath('../../graph_collision_checking_dataset/dataset_2d_7'),
            'mode' : 'validation'}
)

register(
    id='graphEnv2DTest-v7',
    entry_point='sp_rl.envs:GraphEnv2D',
    kwargs={'dataset_folder' : os.path.abspath('../../graph_collision_checking_dataset/dataset_2d_7'),
            'mode' : 'test'}
)

register(
    id='graphEnv2D-v8',
    entry_point='sp_rl.envs:GraphEnv2D',
    kwargs={'dataset_folder' : os.path.abspath('../../graph_collision_checking_dataset/dataset_2d_8'),
            'mode' : 'train',
            'file_idxing' : 0}
)

register(
    id='graphEnv2DValidation-v8',
    entry_point='sp_rl.envs:GraphEnv2D',
    kwargs={'dataset_folder' : os.path.abspath('../../graph_collision_checking_dataset/dataset_2d_8'),
            'mode' : 'validation',
            'file_idxing' : 0}
)

register(
    id='graphEnv2DTest-v8',
    entry_point='sp_rl.envs:GraphEnv2D',
    kwargs={'dataset_folder' : os.path.abspath('../../graph_collision_checking_dataset/dataset_2d_8'),
            'mode' : 'test',
            'file_idxing' : 0}
)