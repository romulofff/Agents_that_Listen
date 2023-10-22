import functools

from gymnasium.spaces import Discrete
from sample_factory.algo.utils.context import (global_env_registry,
                                               global_model_factory)
from sample_factory.envs.env_utils import register_env
from sample_factory.utils.utils import log
from sf_examples.vizdoom.doom.doom_model import make_vizdoom_encoder
# from sf_examples.vizdoom.doom.doom_utils import (DOOM_ENVS, DoomSpec,
#                                                 make_doom_env_from_spec)


from envs.doom.doom_params import add_doom_env_args, doom_override_defaults
from envs.doom.doom_utils import DOOM_ENVS, DoomSpec, make_doom_env_from_spec
from envs.doom.doom_model import make_fft_encoder, make_vizdoom_fft_encoder

# def register_custom_components():
#     global_env_registry().register_env(
#       env_name_prefix='doomsound_',
#       make_env_func=make_doom_env ,
#       add_extra_params_func=add_doom_env_args,
#       override_default_params_func=doom_override_defaults,
#     )


def register_custom_doom_env(custom_timeout=300):
    # absolute path needs to be specified, otherwise Doom will look in the SampleFactory scenarios folder
    # scenario_absolute_path = join(os.path.dirname(__file__), "custom_env", "custom_doom_env.cfg")
    spec = DoomSpec(
        "doom_sound_finder",
        "hell_sound.cfg",  # use your custom cfg here
        Discrete(1+3),
        reward_scaling=0.01,
        default_timeout=custom_timeout,
    )

    # register the env with Sample Factory
    make_env_func = functools.partial(make_doom_env_from_spec, spec)
    register_env(spec.name, make_env_func)


def register_custom_components():
    register_vizdoom_envs()
    register_custom_doom_env()
    register_vizdoom_models()


def register_vizdoom_envs():
    for env_spec in DOOM_ENVS:
        make_env_func = functools.partial(make_doom_env_from_spec, env_spec)
        register_env(env_spec.name, make_env_func)


def register_vizdoom_models():
    global_model_factory().register_encoder_factory(make_vizdoom_fft_encoder)
    # global_model_factory().register_encoder_factory()
