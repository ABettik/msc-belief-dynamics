from pathlib import Path
from lightning.pytorch.cli import LightningCLI
import mlflow

mlflow.set_tracking_uri('file:D:/Arsenii temporrary/arseniis_msc_proj_ray_tune/mlruns_07_01')

# CONFIG_YAML='train.yml'
CONFIG_YAML='train_ndt.yml'

def main() -> None:
    default_conf = Path(__file__).resolve().parent.parent.parent / CONFIG_YAML
    LightningCLI(
        seed_everything_default=42,
        run=True,
        parser_kwargs={
            'parser_mode': 'omegaconf',
            'default_config_files': [str(default_conf)],
        },
        save_config_kwargs={'overwrite': True},
        subclass_mode_model=True,
    )

if __name__ == '__main__':
    main()