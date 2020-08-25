from decentralizedlearning.algs.sac import SACHyperPar

def get_hyperpar(env, alg):
    if env=="HalfCheetah-v2" and alg=="model":
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
                           update_model_every_n_steps=1000,
                           n_steps=30,
                           n_models=10,
                           batch_size=256,
                           weight_decay=0.0,
                           alpha=0.05,
                           use_model=True,
                           use_model_stochastic=True,
                           diverse=True,
                           autotune=True,
                           target_entropy=-3.0)

    elif env=="HalfCheetah-v2" and alg=="SAC":
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
                           target_entropy=-3.0)
    else:
        raise Exception("Config not found")