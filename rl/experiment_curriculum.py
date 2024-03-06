import sys

from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.train import run_rl
from sample_factory.utils.utils import str2bool
#from sf_examples.vizdoom.doom.doom_params import (add_doom_env_args,
#                                                  doom_override_defaults)
from envs.doom.doom_params import add_doom_env_args, doom_override_defaults
from sf_examples.vizdoom.train_vizdoom import parse_vizdoom_cfg
from sample_factory.envs.env_utils import register_env

#from envs.doom.doom_params import add_doom_env_args
# from sample_factory.algo.utils.arguments import default_cfg
from rl.utils.utils import register_custom_components
from envs.doom.doom_utils import make_doom_env


def parse_vizdoom_cfg(argv=None, evaluation=False):
    # print(argv)
    parser, _ = parse_sf_args(argv=argv, evaluation=evaluation)
    # parameters specific to Doom envs
    add_doom_env_args(parser=parser)
    # override Doom default values for algo parameters
    doom_override_defaults(parser=parser)
    # second parsing pass yields the final configuration
    final_cfg = parse_full_cfg(parser, argv)
    return final_cfg


def caso_1_actor_1():
    register_custom_components(sound=True, blind=True)
    args = ["--algo=APPO", 
            "--env=doom_hg_normal_audio_no_poison", 
            "--experiment=actor_1", 
            "--train_for_env_steps=333_333_333", 
            "--num_workers=12", 
            "--num_envs_per_worker=16",
            "--learning_rate=0.001",
            "--batch_size=2048",
            "--exploration_loss_coeff=0.01",
            "--train_dir=/home/romulofff/Documents/sf/Agents_that_Listen/train_dir/curriculum_2/actor_1/"
    ]
    cfg = parse_vizdoom_cfg(argv=args)
    status = run_rl(cfg)
    return status


def caso_2_actor_1():
    register_custom_components(sound=True, blind=True)
    args = ["--algo=APPO", 
            "--env=doom_hg_8med_audio_no_poison", 
            "--experiment=actor_1", 
            "--train_for_env_steps=666_666_666", 
            "--num_workers=12", 
            "--num_envs_per_worker=16",
            "--learning_rate=0.001",
            "--batch_size=2048",
            "--exploration_loss_coeff=0.01",
            "--train_dir=/home/romulofff/Documents/sf/Agents_that_Listen/train_dir/curriculum_2/actor_1/"
    ]
    cfg = parse_vizdoom_cfg(argv=args)
    status = run_rl(cfg)
    return status



def caso_2_actor_2():
    register_custom_components(sound=True, blind=True)
    args = ["--algo=APPO", 
            "--env=doom_hg_8med_audio_no_poison", 
            "--experiment=actor_2", 
            "--train_for_env_steps=666_666_666", 
            "--num_workers=12", 
            "--num_envs_per_worker=16",
            "--learning_rate=0.001",
            "--batch_size=2048",
            "--exploration_loss_coeff=0.01",
            "--train_dir=/home/romulofff/Documents/sf/Agents_that_Listen/train_dir/curriculum_2/actor_2/"
    ]
    cfg = parse_vizdoom_cfg(argv=args)
    status = run_rl(cfg)
    return status

if __name__ == '__main__':
    import shutil
    #main()
    # sys.exit(caso_1())
    caso_1_actor_1()
    shutil.copytree("/home/romulofff/Documents/sf/Agents_that_Listen/train_dir/curriculum_2/actor_1/actor_1", "/media/romulofff/Expansion/Models/BACKUP_CASOS_CURRICULUM/curriculum_2/actor_1/backup/", dirs_exist_ok=True)
    caso_2_actor_1()
    caso_2_actor_2()
    shutil.copytree("/home/romulofff/Documents/sf/Agents_that_Listen/train_dir/curriculum_2/actor_2/actor_2", "/media/romulofff/Expansion/Models/BACKUP_CASOS_CURRICULUM/curriculum_2/actor_2/backup/", dirs_exist_ok=True)

# python -m rl.train --algo=APPO --env=doomsound_instruction --experiment=doom_instruction --encoder_custom=vizdoomSoundFFT --train_for_env_steps=500000000 --num_workers=24 --num_envs_per_worker=20 