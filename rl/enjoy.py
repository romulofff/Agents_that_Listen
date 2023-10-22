import sys

from sample_factory.enjoy import enjoy

from rl.train import register_custom_components, parse_vizdoom_cfg




def main():
    env_name="doom_sound_finder"
#    register_env(env_name,make_doom_env)
    register_custom_components()
    args = ["--algo=APPO", 
            "--env=doom_basic", 
            "--experiment=doom_basic2", 
            "--train_for_env_steps=1000", 
            "--num_workers=10", 
            "--num_envs_per_worker=4",
            "--max_num_episodes=10",
            "--save_video",
            "--video_name=doom_basic2.mp4"]
    #parser = parse_vizdoom_cfg(argv=args)
    # parser = arg_parser()
    #cfg = parse_vizdoom_cfg(argv=["--algo=APPO", f"--env={env_name}", "--experiment=play_doom"])
    cfg = parse_vizdoom_cfg(argv=args, evaluation=True)
    print("CFG LOADED")
    # print(cfg)
    # #cfg = parse_vizdoom_cfg(argv=args)
    status = enjoy(cfg)
    return status

# enjoy_quad
if __name__ == '__main__':
    sys.exit(main())