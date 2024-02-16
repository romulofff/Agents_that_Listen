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


def main():
    env_name="doom_sound_finder"
#    register_env(env_name,make_doom_env)
    register_custom_components()
    args = ["--algo=APPO", 
            "--env=doom_sound_finder", 
            "--experiment=doom_sound_finder", 
            "--train_for_env_steps=100_000_000", 
            "--num_workers=10", 
            "--num_envs_per_worker=4",
            "--render"    
        ]
    #parser = parse_vizdoom_cfg(argv=args)
    # parser = arg_parser()
    #cfg = parse_vizdoom_cfg(argv=["--algo=APPO", f"--env={env_name}", "--experiment=play_doom"])
    cfg = parse_vizdoom_cfg(argv=args)
    print("CFG LOADED")
    # #cfg = parse_vizdoom_cfg(argv=args)
    status = run_rl(cfg)
    return status

def main2():
    env_name="doom_sound_finder"
#    register_env(env_name,make_doom_env)
    register_custom_components()
    args = ["--algo=APPO", 
            "--env=doomsound_instruction", 
            "--experiment=doomsound_instruction", 
            "--train_for_env_steps=500_000_000", 
            "--num_workers=12", 
            "--num_envs_per_worker=12"]
    #parser = parse_vizdoom_cfg(argv=args)
    # parser = arg_parser()
    #cfg = parse_vizdoom_cfg(argv=["--algo=APPO", f"--env={env_name}", "--experiment=play_doom"])
    cfg = parse_vizdoom_cfg(argv=args)
    print("CFG LOADED")
    # #cfg = parse_vizdoom_cfg(argv=args)
    status = run_rl(cfg)
    return status


def main3():
    register_custom_components()
    args = ["--algo=APPO", 
            "--env=doom_basic", 
            "--experiment=doom_basic2", 
            "--train_for_env_steps=10_000_000", 
            "--num_workers=12", 
            "--num_envs_per_worker=12"]
    cfg = parse_vizdoom_cfg(argv=args)
    status = run_rl(cfg)
    return status

def main4():
    register_custom_components()
    args = ["--algo=APPO", 
            "--env=doom_health_gathering_supreme_sound", 
            "--experiment=doom_health_gathering_supreme_sound", 
            "--train_for_env_steps=2_000_000_000", 
            "--num_workers=12", 
            "--num_envs_per_worker=12",
            "--learning_rate=0.0001",
            "--batch_size=2048",
            "--exploration_loss_coeff=0.01"
        ]
    cfg = parse_vizdoom_cfg(argv=args)

    status = run_rl(cfg)
    return status

if __name__ == '__main__':
    #main()
    sys.exit(main4())
# python -m rl.train --algo=APPO --env=doomsound_instruction --experiment=doom_instruction --encoder_custom=vizdoomSoundFFT --train_for_env_steps=500000000 --num_workers=24 --num_envs_per_worker=20 