from decentralizedlearning.algs.sac import SACHyperPar

def get_hyperpar(env, alg):
    if env=="HalfCheetah-v2" and alg=="model":
        return SACHyperPar(hidden_dims_actor=(256, 256),
                           hidden_dims_critic=(256, 256),
                           hidden_dims_model=(200, 200, 200, 200),
                           gamma=0.99,
                           tau=0.005,
                           lr_actor=0.0003,
                           lr_critic=0.0003,
                           lr_model=0.001,
                           l2_norm=0.0,
                           step_random=5000,
                           update_every_n_steps=1,
                           update_model_every_n_steps=250,
                           n_steps=40,
                           n_models=10,
                           batch_size=256,
                           weight_decay=0.0,
                           alpha=0.05,
                           use_model=True,
                           use_model_stochastic=True,
                           diverse=True,
                           autotune=True,
                           target_entropy=-3.0,
                           name=env)
    if env=="HalfCheetah-v2" and alg=="model_degraded":
        return SACHyperPar(hidden_dims_actor=(256, 256),
                           hidden_dims_critic=(256, 256),
                           hidden_dims_model=(200, 200, 200, 200),
                           gamma=0.99,
                           tau=0.005,
                           lr_actor=0.0003,
                           lr_critic=0.0003,
                           lr_model=0.001,
                           l2_norm=0.0,
                           step_random=5000,
                           update_every_n_steps=1,
                           update_model_every_n_steps=250,
                           n_steps=40,
                           n_models=10,
                           batch_size=256,
                           weight_decay=0.0,
                           alpha=0.05,
                           use_model=True,
                           use_model_stochastic=True,
                           diverse=True,
                           autotune=True,
                           target_entropy=-3.0,
                           name=env,
                           use_degraded_sim=True)

    if env=="HalfCheetah-v2" and alg=="model_regulated":
        return SACHyperPar(hidden_dims_actor=(256, 256),
                           hidden_dims_critic=(256, 256),
                           hidden_dims_model=(200, 200, 200, 200),
                           gamma=0.99,
                           tau=0.005,
                           lr_actor=0.0003,
                           lr_critic=0.0003,
                           lr_model=0.001,
                           l2_norm=0.0,
                           step_random=5000,
                           update_every_n_steps=1,
                           update_model_every_n_steps=250,
                           n_steps=40,
                           n_models=10,
                           batch_size=256,
                           weight_decay=0.0,
                           alpha=0.05,
                           use_model=True,
                           use_model_stochastic=True,
                           use_multistep_reg=True,
                           diverse=True,
                           autotune=True,
                           target_entropy=-3.0,
                           name=env)

    if env == "HalfCheetah-v2" and alg == "model_5step":
        return SACHyperPar(hidden_dims_actor=(256, 256),
                           hidden_dims_critic=(256, 256),
                           hidden_dims_model=(200, 200, 200, 200),
                           gamma=0.99,
                           tau=0.005,
                           lr_actor=0.0003,
                           lr_critic=0.0003,
                           lr_model=0.001,
                           l2_norm=0.0,
                           step_random=5000,
                           update_every_n_steps=1,
                           update_model_every_n_steps=250,
                           n_steps=40,
                           n_models=10,
                           batch_size=256,
                           weight_decay=0.0,
                           alpha=0.05,
                           use_model=True,
                           use_model_stochastic=True,
                           use_multistep_reg=True,
                           diverse=True,
                           autotune=True,
                           target_entropy=-3.0,
                           rollout_length=5,
                           name=env)

    if env=="custom" and alg=="SAC":
        return SACHyperPar(hidden_dims_actor=(256, 256),
                           hidden_dims_critic=(256, 256),
                           hidden_dims_model=(200, 200, 200),
                           gamma=0.99,
                           tau=0.005,
                           lr_actor=0.0001,
                           lr_critic=0.0001,
                           lr_model=0.001,
                           l2_norm=0.0,
                           step_random=500,
                           update_every_n_steps=1,
                           update_model_every_n_steps=250,
                           n_steps=1,
                           n_models=10,
                           batch_size=256,
                           weight_decay=0.0,
                           alpha=0.01,
                           use_model=False,
                           use_model_stochastic=True,
                           diverse=True,
                           autotune=False,
                           target_entropy=-1.0,
                           name=env)
    if env=="COM" and alg=="SAC":
        return SACHyperPar(hidden_dims_actor=(128, 128),
                           hidden_dims_critic=(128, 128),
                           hidden_dims_model=(200, 200, 200),
                           gamma=0.95,
                           tau=0.01,
                           lr_actor=0.01,
                           lr_critic=0.01,
                           lr_model=0.01,
                           l2_norm=0.0,
                           step_random=0,
                           update_every_n_steps=100,
                           update_model_every_n_steps=250,
                           n_steps=1,
                           delay=1,
                           n_models=10,
                           batch_size=1024,
                           weight_decay=0.0,
                           alpha=0.005,
                           use_model=False,
                           use_model_stochastic=True,
                           diverse=True,
                           autotune=False,
                           target_entropy=-1.0,
                           name=env,
                           use_common_actor=False,
                           use_common_critic=False)
    if env=="MA" and alg=="SAC":
        return SACHyperPar(hidden_dims_actor=(128, 128),
                           hidden_dims_critic=(128, 128),
                           hidden_dims_model=(200, 200, 200),
                           gamma=0.95,
                           tau=0.01,
                           lr_actor=0.01,
                           lr_critic=0.01,
                           lr_model=0.01,
                           l2_norm=0.0,
                           step_random=0,
                           update_every_n_steps=100,
                           update_model_every_n_steps=250,
                           n_steps=1,
                           delay=1,
                           n_models=10,
                           batch_size=1024,
                           weight_decay=0.0,
                           alpha=0.02,
                           use_model=False,
                           use_model_stochastic=True,
                           diverse=True,
                           autotune=False,
                           target_entropy=-1.0,
                           name=env,
                           use_common_actor=False,
                           use_common_critic=False)
    if env=="MAMODEL" and alg=="SAC":
        return SACHyperPar(hidden_dims_actor=(128, 128),
                           hidden_dims_critic=(256, 256),
                           hidden_dims_model=(200, 200, 200, 200),
                           gamma=0.95,
                           tau=0.01,
                           lr_actor=0.003,
                           lr_critic=0.003,
                           lr_model=0.001,
                           l2_norm=0.0,
                           step_random=00,
                           update_every_n_steps=1,
                           update_model_every_n_steps=1000,
                           n_steps=1,
                           delay=1,
                           n_models=10,
                           batch_size=1024,
                           weight_decay=0.0,
                           alpha=0.02,
                           use_model=True,
                           use_model_stochastic=True,
                           diverse=True,
                           autotune=False,
                           target_entropy=-1.0,
                           name=env,
                           use_common_actor=False,
                           use_common_critic=False,
                           real_ratio=0.1)
    if env=="MAMODELnew" and alg=="SAC":
        return SACHyperPar(hidden_dims_actor=(128, 128),
                           hidden_dims_critic=(256, 256),
                           hidden_dims_model=(200, 200, 200, 200),
                           gamma=0.95,
                           tau=0.01,
                           lr_actor=0.003,
                           lr_critic=0.003,
                           lr_model=0.001,
                           l2_norm=0.0,
                           step_random=00,
                           update_every_n_steps=1,
                           update_model_every_n_steps=1000,
                           n_steps=1,
                           delay=1,
                           n_models=10,
                           batch_size=1024,
                           weight_decay=0.0,
                           alpha=0.02,
                           use_model=False,
                           use_model_stochastic=False,
                           diverse=True,
                           autotune=False,
                           target_entropy=-1.0,
                           name=env,
                           use_common_actor=False,
                           use_common_critic=False,
                           real_ratio=0.1)
    if env=="MAMODEL2" and alg=="SAC":
        return SACHyperPar(hidden_dims_actor=(128, 128),
                           hidden_dims_critic=(128, 128),
                           hidden_dims_model=(200, 200, 200),
                           gamma=0.95,
                           tau=0.01,
                           lr_actor=0.01,
                           lr_critic=0.01,
                           lr_model=0.001,
                           l2_norm=0.0,
                           step_random=00,
                           update_every_n_steps=100,
                           update_model_every_n_steps=25000,
                           n_steps=1,
                           delay=1,
                           n_models=10,
                           batch_size=1024,
                           weight_decay=0.0,
                           alpha=0.02,
                           use_model=False,
                           use_model_stochastic=False,
                           diverse=True,
                           autotune=False,
                           target_entropy=-1.0,
                           name=env,
                           use_common_actor=False,
                           use_common_critic=False,
                           real_ratio=0.1,
                           use_shared_replay_buffer=True)
    if env=="MA2" and alg=="SAC":
        return SACHyperPar(hidden_dims_actor=(128, 128),
                           hidden_dims_critic=(128, 128),
                           hidden_dims_model=(200, 200, 200),
                           gamma=0.95,
                           tau=0.01,
                           lr_actor=0.0005,
                           lr_critic=0.0005,
                           lr_model=0.01,
                           l2_norm=0.0,
                           step_random=1000,
                           update_every_n_steps=5,
                           update_model_every_n_steps=250,
                           n_steps=1,
                           delay=1,
                           n_models=10,
                           batch_size=1024,
                           weight_decay=0.0,
                           alpha=0.002,
                           use_model=False,
                           use_model_stochastic=True,
                           diverse=True,
                           autotune=False,
                           target_entropy=-1.0,
                           name=env,
                           use_common_actor=False,
                           use_common_critic=False,
                           )

    if env=="custom" and alg=="model":
        return SACHyperPar(hidden_dims_actor=(256, 256),
                           hidden_dims_critic=(256, 256),
                           hidden_dims_model=(200, 200, 200),
                           gamma=0.99,
                           tau=0.005,
                           lr_actor=0.001,
                           lr_critic=0.001,
                           lr_model=0.001,
                           l2_norm=0.0,
                           step_random=500,
                           update_every_n_steps=1,
                           update_model_every_n_steps=250,
                           n_steps=20,
                           n_models=10,
                           batch_size=256,
                           weight_decay=0.0,
                           alpha=0.01,
                           use_model=True,
                           use_model_stochastic=True,
                           diverse=True,
                           autotune=False,
                           target_entropy=-1.0,
                           name=env)

    if env=="HalfCheetah-v2" and alg=="model40":
        return SACHyperPar(hidden_dims_actor=(256, 256),
                           hidden_dims_critic=(256, 256),
                           hidden_dims_model=(200, 200, 200, 200),
                           gamma=0.99,
                           tau=0.005,
                           lr_actor=0.0003,
                           lr_critic=0.0003,
                           lr_model=0.001,
                           l2_norm=0.00005,
                           step_random=5000,
                           update_every_n_steps=1,
                           update_model_every_n_steps=250,
                           n_steps=20,
                           n_models=10,
                           batch_size=256,
                           weight_decay=0.0,
                           alpha=0.05,
                           use_model=True,
                           use_model_stochastic=True,
                           diverse=True,
                           autotune=True,
                           target_entropy=-3.0,
                           name=env)

    elif env=="HalfCheetah-v2" and alg=="SAC":
        return SACHyperPar(hidden_dims_actor=(256, 256),
                           hidden_dims_critic=(256, 256),
                           hidden_dims_model=(200, 200, 200),
                           gamma=0.99,
                           tau=0.005,
                           lr_actor=0.0003,
                           lr_critic=0.0003,
                           lr_model=0.001,
                           l2_norm=0.00005,
                           step_random=5000,
                           update_every_n_steps=1,
                           update_model_every_n_steps=1000,
                           n_steps=1,
                           n_models=10,
                           batch_size=256,
                           weight_decay=0.0,
                           alpha=0.05,
                           use_model=False,
                           use_model_stochastic=True,
                           diverse=True,
                           autotune=True,
                           target_entropy=-3.0,
                           name=env)

    if env=="Hopper-v2" and alg=="model":
        return SACHyperPar(hidden_dims_actor=(256, 256),
                           hidden_dims_critic=(256, 256),
                           hidden_dims_model=(200, 200, 200),
                           gamma=0.99,
                           tau=0.005,
                           lr_actor=0.0003,
                           lr_critic=0.0003,
                           lr_model=0.001,
                           l2_norm=0.0,
                           step_random=5000,
                           update_every_n_steps=1,
                           update_model_every_n_steps=250,
                           n_steps=20,
                           n_models=7,
                           batch_size=256,
                           weight_decay=0.0,
                           alpha=0.05,
                           use_model=True,
                           use_model_stochastic=True,
                           diverse=True,
                           autotune=True,
                           target_entropy=-1.0,
                           name=env)

    if env=="Hopper-v2" and alg=="SAC":
        return SACHyperPar(hidden_dims_actor=(256, 256),
                           hidden_dims_critic=(256, 256),
                           hidden_dims_model=(200, 200, 200),
                           gamma=0.99,
                           tau=0.005,
                           lr_actor=0.0003,
                           lr_critic=0.0003,
                           lr_model=0.001,
                           l2_norm=0.0,
                           step_random=5000,
                           update_every_n_steps=1,
                           update_model_every_n_steps=250,
                           n_steps=1,
                           n_models=7,
                           batch_size=256,
                           weight_decay=0.0,
                           alpha=0.05,
                           use_model=False,
                           use_model_stochastic=True,
                           diverse=True,
                           autotune=True,
                           target_entropy=-1.0,
                           name=env)

    else:
        raise Exception("Config not found")