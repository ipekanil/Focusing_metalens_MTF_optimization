__author__ = "Anil Appak"
__email__ = "ipekanilatalay@gmail.com"
__organization__ = "Tampere University"

from dataclasses import dataclass
import os
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from OpticsOnlyLit import OpticsOnlyLit


class JobType:
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass
class Config:
    entity: str = "ipekanil"
    project: str = "focusing_metalens_optimization"
    job_type: str = JobType.TRAIN
    tags: list = None

    seed: int = 42
    deterministic: bool = False
    accelerator: str = "gpu"
    devices: int = 1
    strategy: str = "auto"
    num_nodes: int = 1

    checkpoint_dir: str = "checkpoints_optics_only"
    save_top_k: int = -1
    every_n_epochs: int = 1

    mat_files_directory: str = "./meta_data"

    wavelengths_m: list = None
    N: int = 5000 #simulation window size in pixels, reduces boundary artifacts in the PSF, use a larger grid size than the metalens diameter. Ex: The metalens diameter is 0.5 mm, so with dx=500 nm, we need at least 1000 pixels to cover the metalens, and we add some margin for better PSF quality.
    dx_m: float = 225e-9 #metalens sampling interval in meters per pixel, also the pixel size of the radius map.
    focal_length_m: float = 8e-3 #focal length in meters, used for focusing phase. 
    metalens_diameter_m: float = 1e-3 #metalens diameter in meters, should be less than or equal to N * dx_m (grid size), otherwise it will raise an error.

    if metalens_diameter_m > N * dx_m:
            raise ValueError(f"Metalens diameter {metalens_diameter_m} m exceeds grid size {N * dx_m} m. Please reduce metalens diameter or increase grid size.") 

    n_z: int = 1 #number of depth planes to simulate, currently only supports 1 (focusing plane), but can be increased for EDOF optimization.
    z_span_m: float = 0.0 #span of depth planes in meters, only used if n_z > 1. For example, if n_z=3 and z_span_m=0.2, it will simulate depth planes at -0.1 m, 0 m, and +0.1 m relative to the focal plane.

    epoch: int = 200
    lr_optical: float = 5e-2

    rho_cut: float = 0.4 #normalized spatial frequency cutoff for MTF calculation, between 0 and 1. This determines the maximum spatial frequency considered in the MTF loss. A value of 0.4 means that spatial frequencies above 40% of the Nyquist frequency will be ignored in the MTF loss calculation.
    w_focus: float = 0.5 #weight for the centroid loss, which encourages the PSF to be centered at the focal point. This is a value between 0 and 1 that determines how much importance to give to the centroid loss relative to the MTF loss. A higher value means the optimization will focus more on centering the PSF, while a lower value means it will focus more on improving the MTF. The optimal value may require some experimentation, but starting with 0.2 is a reasonable choice to balance both objectives.
    w_mtf: float = 0.5 #weight for the M2 loss, which encourages the PSF to be more compact and have less energy in the side lobes. This is a value between 0 and 1 that determines how much importance to give to the M2 loss relative to the MTF loss. A higher value means the optimization will focus more on reducing the M2 factor of the PSF, while a lower value means it will focus more on improving the MTF. The optimal value may require some experimentation, but starting with 0.0 means that initially we will not use the M2 loss and focus solely on the MTF and centroid losses.
    use_worst_depth: bool = True


def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    config = Config()
    if config.tags is None:
        config.tags = ["optics_only", "mtf"]

    pl.seed_everything(config.seed, workers=True)

    if config.wavelengths_m is None:
        config.wavelengths_m = [681e-9, 601e-9, 521e-9, 433e-9]

    logger = None
    try:
        logger = WandbLogger(
            entity=config.entity,
            project=config.project,
            log_model="all",
            tags=config.tags,
        )
        logger.experiment.config.update(vars(config), allow_val_change=True)
    except Exception as e:
        print("WandB logger init failed:", e)

    model = OpticsOnlyLit(
        mat_files_directory=config.mat_files_directory,
        wavelengths_m=config.wavelengths_m,
        N=config.N,
        dx_m=config.dx_m,
        focal_length_m=config.focal_length_m,
        metalens_diameter_m=config.metalens_diameter_m,
        n_z=config.n_z,
        z_span_m=config.z_span_m,
        lr_optical=config.lr_optical,
        rho_cut=config.rho_cut,
        w_focus=config.w_focus,
        w_mtf=config.w_mtf,
        use_worst_depth=config.use_worst_depth,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=config.checkpoint_dir,
            save_top_k=config.save_top_k,
            every_n_epochs=config.every_n_epochs,
            verbose=True,
        )
    ]

    trainer = pl.Trainer(
        max_epochs=config.epoch,
        callbacks=callbacks,
        logger=logger,
        deterministic=config.deterministic,
        devices=config.devices,
        accelerator=config.accelerator,
        strategy=config.strategy,
        num_nodes=config.num_nodes,
        log_every_n_steps=1,
    )

    trainer.fit(model)


if __name__ == "__main__":
    main()
