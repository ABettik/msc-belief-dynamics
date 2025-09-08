import ray
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from lightning.pytorch.cli import LightningCLI
from ray.tune.schedulers import ASHAScheduler
from ray.air import session
from lightning.pytorch.loggers import MLFlowLogger
from ray.air.config import CheckpointConfig

CONFIG_YAML = "C:/Users/qf1824/Desktop/coding/Arseniis_Msc_proj/tune_ndt.yml"

def train_tune(config):
    trial_id   = session.get_trial_id()
    # print(config)
    run_name   = f"trial_{trial_id[:8]}_ld{config['latent_dim']}_lr{config['lr']:.1e}"
    # return
    overrides = {
        
        'model': {
            'init_args': {
                'latent_dim': config['latent_dim'],
                
                'optimizer_args': { 'lr': config['lr'], 'weight_decay': config['weight_decay'] },
                'lr_scheduler_args': { 'max_lr': config['lr'] },
                
                'ndt_cfg': {
                    'encoder': {
                        'transformer': {
                            'hidden_size': config['latent_dim'],
                            'n_layers': config['transformer_layers'],
                        },
                        # 'masker': {
                        #     'ratio': config['mask_ratio'],
                        # },
                    },
                },
                # 'tc_weight': config['tc_weight'],
                # 'prpd_weight': config['prpd_weight'],
                # 'mlm_weight': config['mlm_weight'],
                
                # 'tc_warmup_frac': config['tc_warmup_frac'],
                'd_lr': config['d_lr'],
                # 'd_update_every': config['d_update_every'],
                # 'perm_repeats': config['perm_repeats'],
                'r1_gamma': config['r1_gamma'],
            }
        }
        # },
    }

    cli = LightningCLI(
        seed_everything_default=378,
        run=False,
        args=overrides,
        parser_kwargs={
            "parser_mode": "omegaconf",
            "default_config_files": [CONFIG_YAML],
        },
        save_config_callback=None,
        subclass_mode_model=True,
    )
    cli.trainer.logger = MLFlowLogger(
        experiment_name="tune_NDT1SSL_12_08_masker_fix",
        tracking_uri='file:D:/Arsenii temporrary/arseniis_msc_proj_ray_tune/mlruns_07_01',
        run_name=run_name,
        log_model=False,
    )

    # report metrics & save checkpoints back to Ray
    cli.trainer.callbacks.append(
        TuneReportCheckpointCallback(
            # metrics={"val_loss": "val/prpd_r2"},
            metrics={"val_loss": "val/loss/total"},
            filename="checkpoint",
            on="validation_end",
            save_checkpoints=False
        )
    )
    # trainer = prepare_trainer(cli.trainer)
    # trainer.fit(cli.model, datamodule=cli.datamodule)
    cli.trainer.fit(cli.model, cli.datamodule)


if __name__ == "__main__":
    ray.init()
    tuner = tune.Tuner(
        tune.with_resources(train_tune, resources={"cpu": 12, "gpu": 1}),

        param_space={
            "latent_dim": tune.choice([16, 128]),
            "transformer_layers": tune.choice([2, 6]),
            
            "lr":  tune.loguniform(1e-4, 1e-3),
            "weight_decay": tune.choice([0.1, 0.01, 0.001, 0.0001]),
            
            # "tc_warmup_frac": tune.choice([0.05, 0.1, 0.2]),
            "d_lr": tune.loguniform(5e-5, 2e-4),
            # "d_update_every": tune.choice([1,2,4]),
            # "perm_repeats": tune.choice([1,2]),
            "r1_gamma": tune.choice([0.0, 5.0]),
        },

        tune_config=tune.TuneConfig(
            metric="val_loss",
            mode="min",
            num_samples = 1000,
            max_concurrent_trials = 1,
            trial_dirname_creator=lambda trial: f"{trial.trial_id}",
            scheduler=ASHAScheduler(
                max_t=300,
                grace_period=100,
                reduction_factor=3
            )
        ), 
        run_config=tune.RunConfig(
            name="ndt1_mtm_mfix_12082025",
            storage_path="D:/Arsenii temporrary/arseniis_msc_proj_ray_tune",
            checkpoint_config=CheckpointConfig(num_to_keep=1, checkpoint_frequency=0)
        ),
    )
    tuner.fit()