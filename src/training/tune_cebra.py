import ray
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from lightning.pytorch.cli import LightningCLI
from ray.tune.schedulers import ASHAScheduler
from ray.air import session
from lightning.pytorch.loggers import MLFlowLogger
from ray.air.config import CheckpointConfig

CONFIG_YAML = "C:/Users/qf1824/Desktop/coding/Arseniis_Msc_proj/tune.yml"

def train_tune(config):
    trial_id   = session.get_trial_id()
    run_name   = f"trial_{trial_id[:8]}_ld{config['latent_dim']}_lr{config['lr']:.1e}"
    
    overrides = {
        'model': {
            'lr': config['lr'],
            'latent_dim': config['latent_dim'],
            'num_hidden_units': config['num_hidden_units'],
            'temperature': config['temperature'],
            'tc_weight': config['tc_weight'],
            'prpd_weight': config['prpd_weight'],
            'prpd_head_hidden_dims': config['prpd_head_hidden_dims'],
            'recon_weight': config['recon_weight'],
        },
    }

    cli = LightningCLI(
        seed_everything_default=42,
        run=False,
        args=overrides,
        parser_kwargs={
            "parser_mode": "omegaconf",
            "default_config_files": [CONFIG_YAML],
        },
        save_config_callback=None,
    )
    cli.trainer.logger = MLFlowLogger(
        experiment_name="tune_CEBRA_Factor_Reg_01_08",
        tracking_uri='file:D:/Arsenii temporrary/arseniis_msc_proj_ray_tune/mlruns_07_01',
        run_name=run_name,
        log_model=False,
    )

    # report metrics and save checkpoints back to Ray
    cli.trainer.callbacks.append(
        TuneReportCheckpointCallback(
            metrics={"val_loss": "val/prpd_r2"},
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
        tune.with_resources(train_tune,
                            resources={"cpu": 12, "gpu": 1}),
        param_space={
            # "lr":              tune.loguniform(1e-4, 5e-3),
            # "latent_dim":      tune.choice([8, 16, 32, 64]),
            # "num_hidden_units":tune.choice([400, 800, 1200]),
            # "temperature":     tune.loguniform(0.03, 0.2),
            # "tc_weight":       tune.loguniform(1.0, 20.0),
            # "prpd_weight":     tune.loguniform(0.5, 5.0),
            # "prpd_head_hidden_dims": tune.choice([[32], [64, 64]]),
            # "recon_weight":    tune.choice([0.0, 0.05, 0.1]),
            
            "lr":              tune.loguniform(2e-4, 2e-3),
            "latent_dim":      tune.choice([16, 64]),
            "num_hidden_units":tune.choice([800, 1000]),
            "temperature":     tune.loguniform(0.05, 0.15),
            "tc_weight":       tune.loguniform(1.0, 4.0),
            "prpd_weight":     tune.loguniform(0.7, 2.1),
            "prpd_head_hidden_dims": tune.choice([[32], [64, 64]]),
            "recon_weight":    tune.choice([0.05, 0.1]),
        },

        tune_config=tune.TuneConfig(
            metric="val_loss",
            mode="max",
            num_samples = 100,
            max_concurrent_trials = 1,
            trial_dirname_creator=lambda trial: f"{trial.trial_id}",
            scheduler=ASHAScheduler(
                max_t=150,
                grace_period=10,
                reduction_factor=3
            )
        ), 
        run_config=tune.RunConfig(
            name="cebra_01_08_2025",
            storage_path="D:/Arsenii temporrary/arseniis_msc_proj_ray_tune",
            checkpoint_config=CheckpointConfig(num_to_keep=1, checkpoint_frequency=0)
        ),
    )
    tuner.fit()