# visualize_trained_optics.py
# Load a trained OpticsOnlyLit checkpoint and save PSF, MTF, phase, and radius map images.

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from OpticsOnlyLit import OpticsOnlyLit


def rotate_radius_vector_nearest(radius_vec, N, center=None):
    device = radius_vec.device
    dtype = radius_vec.dtype
    R = radius_vec.numel()

    if center is None:
        cy = cx = (N - 1) / 2.0
    else:
        cy, cx = center

    yy = torch.arange(N, device=device, dtype=dtype)
    xx = torch.arange(N, device=device, dtype=dtype)
    Y, X = torch.meshgrid(yy, xx, indexing="ij")
    r_pix = torch.sqrt((Y - cy) ** 2 + (X - cx) ** 2)

    ring_idx = torch.clamp(r_pix.round().long(), 0, R - 1)
    return radius_vec[ring_idx]

def save_im(path, img2d, title=None, cmap="gray", vmin=None, vmax=None):
    plt.figure()
    plt.imshow(img2d, cmap=cmap, vmin=vmin, vmax=vmax)
    if title is not None:
        plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def fft_mtf(psf2d):
    # psf2d: torch [H,W], nonnegative, not necessarily normalized
    psf2d = psf2d.clamp_min(0)
    psf2d = psf2d / (psf2d.sum() + 1e-12)
    mtf = torch.fft.fftshift(torch.fft.fft2(psf2d, norm="backward"))
    mtf_mag = mtf.abs()
    H, W = mtf_mag.shape
    cy, cx = H // 2, W // 2
    dc = mtf_mag[cy, cx].clamp_min(1e-8)
    return mtf_mag / dc


def main():
    ckpt_path = "./checkpoints_optics_only/epoch=199-step=200.ckpt"
    out_dir = "./viz_outputs"
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lit = OpticsOnlyLit.load_from_checkpoint(ckpt_path, map_location=device, strict=False)
    lit.eval()
    lit.to(device)

    with torch.no_grad():
        psf_list, radius_map, radius_vector, aperture_mask, phase_map, radius_map_init = lit.optic()

    radius_map_cpu = (radius_map * aperture_mask).detach().cpu().float().numpy()
    aperture_cpu = aperture_mask.detach().cpu().float().numpy()

    save_im(
        os.path.join(out_dir, "radius_map_nm.png"),
        radius_map_cpu * 1e9,
        title="Radius map (nm)",
        cmap="viridis",
    )

    save_im(
        os.path.join(out_dir, "aperture_mask.png"),
        aperture_cpu,
        title="Aperture mask",
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )

    radius_map_init = rotate_radius_vector_nearest(radius_map_init, lit.optic.N, center=None)
    radius_map_init_cpu = (radius_map_init * aperture_mask).detach().cpu().float().numpy()


    save_im(
        os.path.join(out_dir, "radius_map_init_nm.png"),
        radius_map_init_cpu * 1e9,
        title="Radius map init (nm)",
        cmap="viridis",
    )

    radius_map_diff_cpu = (radius_map - radius_map_init) * aperture_mask
    radius_map_diff_cpu = radius_map_diff_cpu.detach().cpu().float().numpy()
    save_im(
        os.path.join(out_dir, "radius_map_diff_nm.png"),
        radius_map_diff_cpu * 1e9,
        title="Radius map diff (nm)",
        cmap="bwr",
    )

    
    wavelengths_m = None
    try:
        wavelengths_m = list(lit.hparams.wavelengths_m)
    except Exception:
        wavelengths_m = [681e-9, 601e-9, 521e-9, 433e-9]

        # Phase map saving (supports both tensor and list-of-tensors)
    if phase_map is not None:
        phase_out_dir = os.path.join(out_dir, "phase_maps")
        os.makedirs(phase_out_dir, exist_ok=True)

        def wrap_to_pi(x):
            return ((x + torch.pi) % (2 * torch.pi)) - torch.pi

        if isinstance(phase_map, (list, tuple)):
            for i, ph in enumerate(phase_map):
                lam_nm = int(round(wavelengths_m[i] * 1e9)) if wavelengths_m is not None else i
                ph2d = (ph * aperture_mask).detach().float()
                ph2d = wrap_to_pi(ph2d)
                ph_cpu = ph2d.cpu().numpy()

                save_im(
                    os.path.join(phase_out_dir, f"phase_{lam_nm}nm.png"),
                    ph_cpu,
                    title=f"Phase map (rad) {lam_nm} nm",
                    cmap="twilight",
                    vmin=-np.pi,
                    vmax=np.pi,
                )
        else:
            ph2d = (phase_map * aperture_mask).detach().float()
            ph2d = wrap_to_pi(ph2d)
            ph_cpu = ph2d.cpu().numpy()

            save_im(
                os.path.join(out_dir, "phase_map_rad.png"),
                ph_cpu,
                title="Phase map (rad)",
                cmap="twilight",
                vmin=-np.pi,
                vmax=np.pi,
            )


    for c, psf_c in enumerate(psf_list):
        # psf_c: [D,1,H,W]
        if psf_c.dim() == 4 and psf_c.shape[1] == 1:
            psf_c = psf_c[:, 0]  # [D,H,W]

        D, H, W = psf_c.shape
        lam_nm = int(round(wavelengths_m[c] * 1e9))

        for d in range(D):
            psf2d = psf_c[d].detach().float().cpu()
            psf2d_np = psf2d.numpy()

            psf_path = os.path.join(out_dir, f"psf_{lam_nm}nm_z{d:02d}.png")
            save_im(psf_path, psf2d_np, title=f"PSF {lam_nm} nm  z index {d}", cmap="gray")

            mtf2d = fft_mtf(psf2d.to(device)).detach().cpu().float().numpy()
            mtf_path = os.path.join(out_dir, f"mtf_{lam_nm}nm_z{d:02d}.png")
            save_im(mtf_path, mtf2d, title=f"MTF {lam_nm} nm  z index {d}", cmap="gray", vmin=0.0, vmax=1.0)

            cy, cx = H // 2, W // 2
            row = psf2d_np[cy, :]
            col = psf2d_np[:, cx]

            plt.figure()
            plt.plot(row)
            plt.title(f"PSF center row {lam_nm} nm  z index {d}")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"psf_row_{lam_nm}nm_z{d:02d}.png"), dpi=200)
            plt.close()

            plt.figure()
            plt.plot(col)
            plt.title(f"PSF center col {lam_nm} nm  z index {d}")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"psf_col_{lam_nm}nm_z{d:02d}.png"), dpi=200)
            plt.close()

    rv = radius_vector.detach().cpu().float().numpy()
    print("Saved visualizations to:", os.path.abspath(out_dir))
    print("radius_vector nm min max:", float(rv.min() * 1e9), float(rv.max() * 1e9))


if __name__ == "__main__":
    main()
