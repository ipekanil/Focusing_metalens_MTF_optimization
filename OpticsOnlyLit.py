import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import math

from OpticsModel import OpticsModel


class _OneBatchDataset(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return torch.tensor(0)


def _radial_profile_via_polar(mtf_2d, n_r=128, n_theta=64):
    H, W = mtf_2d.shape
    device, dtype = mtf_2d.device, mtf_2d.dtype
    r = torch.linspace(0, 1, n_r, device=device, dtype=dtype)
    th = torch.linspace(0, 2 * math.pi, n_theta, device=device, dtype=dtype)
    R, TH = torch.meshgrid(r, th, indexing="ij")
    grid = torch.stack([R * torch.cos(TH), R * torch.sin(TH)], dim=-1).unsqueeze(0)
    img = mtf_2d.unsqueeze(0).unsqueeze(0)
    polar = F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    prof = polar[0, 0].mean(dim=-1)
    return r, prof


def edof_mtf_rmse_simple(psf_list, rho_cut=0.7, use_worst_depth=True, return_per_channel=False):
    n_r, n_theta, alpha = 128, 64, 40.0

    ref = psf_list[0] if isinstance(psf_list, (list, tuple)) else psf_list
    device, dtype = ref.device, ref.dtype
    rho = torch.linspace(0, 1, n_r, device=device, dtype=dtype)

    edge = 0.05
    target = (rho <= rho_cut - edge).float()
    t = ((rho - (rho_cut - edge)) / (2 * edge)).clamp(0, 1)
    target = target + 0.5 * (1 - torch.cos(math.pi * t)) * ((rho > rho_cut - edge) & (rho < rho_cut + edge)).float()
    target = target.clamp(max=1.0)

    weight = (rho <= rho_cut).float()
    weight[rho < 0.02] = 0.0
    weight = weight * (rho / (rho_cut + 1e-12)).pow(1.5)
    weight = weight / (weight.sum() + 1e-12)

    per_channel_val = []
    for psf_c in psf_list:
        if psf_c.dim() == 4 and psf_c.shape[1] == 1:
            psf_c = psf_c.squeeze(1)
        psf_c = psf_c.clamp_min(0)
        psf_c = psf_c / (psf_c.sum(dim=(-1, -2), keepdim=True) + 1e-12)

        D, H, W = psf_c.shape

        mtf = torch.fft.fftshift(torch.fft.fft2(psf_c, norm="backward"), dim=(-2, -1))
        mtf_mag = mtf.abs()
        cy, cx = mtf_mag.shape[-2] // 2, mtf_mag.shape[-1] // 2
        dc = mtf_mag[..., cy, cx].clamp_min(1e-8).unsqueeze(-1).unsqueeze(-1)
        mtf_mag = mtf_mag / dc

        profs = []
        for d in range(D):
            _, p = _radial_profile_via_polar(mtf_mag[d], n_r=n_r, n_theta=n_theta)
            profs.append(p)
        profs = torch.stack(profs, dim=0)

        huber = ((profs - target).abs() * weight).sum(dim=1)

        if use_worst_depth:
            huber = torch.logsumexp(alpha * huber, dim=0) / alpha
        else:
            huber = huber.mean()

        per_channel_val.append(huber)

    per_channel_loss = torch.stack(per_channel_val, dim=0)
    gamma = 30.0  # bigger -> closer to max()
    loss = torch.logsumexp(gamma * per_channel_loss, dim=0) / gamma

    #loss = per_channel_loss.mean()
    logs = {"edof/rmse": loss}
    return per_channel_loss, loss, logs


import torch

def encircled_energy_loss(psf_list, r0_px=8, reduce_depth="mean", reduce_channel="mean"):
    """
    psf_list: list of psf_c, each psf_c is [D,1,H,W] or [D,H,W]
    r0_px: radius of the encircled energy circle in pixels
    Returns: loss scalar, per_channel tensor
    """

    per_ch = []
    for psf_c in psf_list:
        if psf_c.dim() == 4 and psf_c.shape[1] == 1:
            psf_c = psf_c[:, 0]  # [D,H,W]

        psf_c = psf_c.clamp_min(0)
        psf_c = psf_c / (psf_c.sum(dim=(-1, -2), keepdim=True) + 1e-12)

        D, H, W = psf_c.shape
        y = torch.arange(H, device=psf_c.device, dtype=psf_c.dtype)
        x = torch.arange(W, device=psf_c.device, dtype=psf_c.dtype)
        Y, X = torch.meshgrid(y, x, indexing="ij")

        cy = (H - 1) / 2.0
        cx = (W - 1) / 2.0
        r2 = (X - cx) ** 2 + (Y - cy) ** 2
        mask = (r2 <= float(r0_px * r0_px)).to(psf_c.dtype)  # [H,W]

        ee = (psf_c * mask[None, :, :]).sum(dim=(-1, -2))  # [D]
        loss_d = 1.0 - ee  # minimize this

        if reduce_depth == "min":
            loss_c = loss_d.min()
        else:
            loss_c = loss_d.mean()

        per_ch.append(loss_c)

    per_ch = torch.stack(per_ch, dim=0)  # [C]

    if reduce_channel == "max":
        loss = per_ch.max()
    else:
        loss = per_ch.mean()

    return loss, per_ch



def psf_second_moment(psf_list, reduce_depth="min"):
    total = 0.0
    n = 0
    for psf_c in psf_list:
        if psf_c.dim() == 4 and psf_c.shape[1] == 1:
            psf_c = psf_c[:, 0]
        psf_c = psf_c.clamp_min(0)
        psf_c = psf_c / (psf_c.sum(dim=(-1, -2), keepdim=True) + 1e-12)

        D, H, W = psf_c.shape
        y = torch.linspace(-(H - 1) / 2, (H - 1) / 2, H, device=psf_c.device, dtype=psf_c.dtype)
        x = torch.linspace(-(W - 1) / 2, (W - 1) / 2, W, device=psf_c.device, dtype=psf_c.dtype)
        Y, X = torch.meshgrid(y, x, indexing="ij")
        r2 = X * X + Y * Y
        m2 = (psf_c * r2).sum(dim=(-1, -2))

        if reduce_depth == "min":
            total = total + m2.min()
        elif reduce_depth == "mean":
            total = total + m2.mean()
        else:
            total = total + m2.min()
        n += 1

    return total / max(1, n)


class OpticsOnlyLit(pl.LightningModule):
    def __init__(
        self,
        mat_files_directory,
        wavelengths_m,
        N,
        dx_m,
        focal_length_m,
        metalens_diameter_m,
        n_z=1,
        z_span_m=0.0,
        lr_optical=1e-1,
        w_mtf=1.0,
        w_focus=1.0,
        rho_cut=0.4,
        use_worst_depth=True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self._prev_radius_vector = None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.optic = OpticsModel(
            mat_files_directory=mat_files_directory,
            wavelengths_m=wavelengths_m,
            N=N,
            dx_m=dx_m,
            focal_length_m=focal_length_m,
            metalens_diameter_m=metalens_diameter_m,
            n_z=n_z,
            z_span_m=z_span_m,
            device=device,
        )

    def train_dataloader(self):
        return DataLoader(_OneBatchDataset(), batch_size=1, num_workers=0)

    def configure_optimizers(self):
        return torch.optim.Adam([self.optic.radius_vector_learnable], lr=float(self.hparams.lr_optical))

    def training_step(self, batch, batch_idx):
        psf_list, radius_map, radius_vector, aperture_mask, phase_map, radius_map_init = self.optic()

        with torch.no_grad():
            rv = radius_vector.detach()

            if self._prev_radius_vector is None:
                delta_mean = torch.tensor(0.0, device=self.device)
                delta_max = torch.tensor(0.0, device=self.device)
            else:
                delta = (rv - self._prev_radius_vector).abs()
                delta_mean = delta.mean()
                delta_max = delta.max()

            self._prev_radius_vector = rv.clone()


        per_ch, mtf_loss, _ = edof_mtf_rmse_simple(
            psf_list,
            rho_cut=float(self.hparams.rho_cut),
            use_worst_depth=bool(self.hparams.use_worst_depth),
            return_per_channel=True,
        )

        focus_loss, focus_per_ch = encircled_energy_loss(
            psf_list,
            r0_px=8,                 # start with 6 to 12
            reduce_depth="mean",      # D is 1 if you fix z
            reduce_channel="max",     # pushes the worst wavelength
        )

        loss = float(self.hparams.w_mtf) * mtf_loss + float(self.hparams.w_focus)* focus_loss 

        grad = self.optic.radius_vector_learnable.grad
        if grad is None:
            gnorm = torch.tensor(0.0, device=self.device)
        else:
            gnorm = grad.detach().norm()

        self.log("train/radius_grad_norm", gnorm, on_step=True, on_epoch=False, prog_bar=True)

        self.log("train/loss_total", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/mtf_loss", mtf_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/focus_loss", focus_loss, on_step=True, on_epoch=True, prog_bar=True)

        if isinstance(per_ch, torch.Tensor) and per_ch.numel() == len(psf_list):
            for i, lam in enumerate(self.hparams.wavelengths_m):
                self.log(f"train/mtf_{int(lam*1e9)}", per_ch[i], on_step=True, on_epoch=True)

        self.log("train/radius_nm_min", (radius_vector.min() * 1e9), on_step=True, on_epoch=True)
        self.log("train/radius_nm_max", (radius_vector.max() * 1e9), on_step=True, on_epoch=True)
        self.log("train/radius_delta_mean_nm", delta_mean * 1e9, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train/radius_delta_max_nm", delta_max * 1e9, on_step=True, on_epoch=False, prog_bar=False)

        return loss
