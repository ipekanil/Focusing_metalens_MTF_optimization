import torch
import torch.nn.functional as F
import numpy as np
import h5py, os

##############################################################################
###Helper functions for optics simulation and radius map manipulation######
##############################################################################

def propagate_angularspec_fast(uin, M, lambda_m, dx, z, use_gpu=True, bandlimit=True):
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    uin = uin.to(device)

    k = 2 * np.pi / lambda_m

    fx = torch.fft.fftfreq(M, d=dx, device=device)
    fx = torch.fft.fftshift(fx)
    FX, FY = torch.meshgrid(fx, fx, indexing="ij")

    sqrt_arg = 1 - (lambda_m * FX) ** 2 - (lambda_m * FY) ** 2

    if bandlimit:
        pass_mask = (sqrt_arg >= 0).to(torch.float32)
        sqrt_arg = torch.clamp(sqrt_arg, min=0.0)
    else:
        pass_mask = 1.0

    H = torch.exp(1j * k * z * torch.sqrt(sqrt_arg.to(torch.complex64))) * pass_mask

    U1 = torch.fft.fftshift(torch.fft.fft2(uin))
    U2 = U1 * H
    U2 = torch.fft.ifftshift(U2)
    uout = torch.fft.ifft2(U2)
    return uout

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


def circular_aperture_mask(N, radius_px, device):
    yy = torch.arange(N, device=device, dtype=torch.float32)
    xx = torch.arange(N, device=device, dtype=torch.float32)
    Y, X = torch.meshgrid(yy, xx, indexing="ij")
    cy = cx = (N - 1) / 2.0
    r = torch.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    return (r <= float(radius_px)).to(torch.float32)


##############################################################################
###Optics Model######
##############################################################################
class OpticsModel(torch.nn.Module):
    def __init__(
        self,
        mat_files_directory,
        wavelengths_m,
        N, #simulation window size in pixels
        dx_m, #sampling interval in meters at the metalens plane, also the pixel size of the radius map
        focal_length_m,
        metalens_diameter_m, #metalens diameter in meters, L = N * dx_m, metalens_diameter_m ≤ L (grid size)
        n_z=1,
        z_span_m=0.0,
        device=None,
    ):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.wavelengths = torch.tensor(wavelengths_m, dtype=torch.float32, device=self.device)
        self.n_lambda = int(len(wavelengths_m))

        self.N = int(N)
        self.dx = float(dx_m)

        self.f = float(focal_length_m)
        self.n_z = int(n_z)
        self.z_span = float(z_span_m)

        self.metalens_diameter = float(metalens_diameter_m)
        if self.metalens_diameter > self.N * self.dx:
            raise ValueError(f"Metalens diameter {self.metalens_diameter} m exceeds grid size {self.N * self.dx} m. Please reduce metalens diameter or increase grid size.")

        radius_px = (self.metalens_diameter * 0.5) / self.dx #aperture diameter in pixels
        self.aperture_mask = circular_aperture_mask(self.N, radius_px, self.device)


        file_path_radius = os.path.join(mat_files_directory, "edge_PRGB_TiO2_225P_1um_h.mat")
        with h5py.File(file_path_radius, "r") as f:
            self.radius_values = torch.tensor(f["edge"][:], dtype=torch.float32, device=self.device)

        file_path_phase = os.path.join(mat_files_directory, "phase_PRGB_TiO2_225P_1um_h.mat")
        with h5py.File(file_path_phase, "r") as f:
            self.phase_broadband = torch.tensor(f["phase_matrix"][:], dtype=torch.float32, device=self.device)

        self.radius_min = 100e-9 #minimum radius in meters, value from the metalens library
        self.radius_max = 175e-9 #maximum radius in meters, value from the metalens library
        self.SCALE = 0.2 #scaling factor for the learnable parameterization, adjust based on the expected radius range and optimization stability. A smaller SCALE means the optimization will take smaller steps in the radius space, which can improve stability but may require more iterations.

        R_pixel_num = self.N // 2

        """
        ################################
        #Flat pi phase initialization
        ################################
        flat_pi_phase_vector = torch.full((R_pixel_num,), np.pi, device=self.device, dtype=torch.float32)
        init_r = self.differentiable_interp1d(
            self.phase_broadband[2],
            self.radius_values.squeeze(),
            flat_pi_phase_vector,
        )
        init_r = init_r.clamp(self.radius_min + 1e-12, self.radius_max - 1e-12)
        p = (init_r - self.radius_min) / (self.radius_max - self.radius_min)
        logits0 = torch.logit(p) / self.SCALE
        self.radius_vector_learnable = torch.nn.Parameter(logits0, requires_grad=True)
        """

        ########################################################
        # Focusing phase initialization at one reference wavelength
        ########################################################
        ref_idx = 2  # for example 601 nm, adjust to your wavelength ordering
        lambda_ref = float(self.wavelengths[ref_idx])  # meters
        # radial coordinate in meters for the 1D vector
        # IMPORTANT: dx here must be the sampling at the metalens plane in meters per pixel
        r = torch.arange(R_pixel_num, device=self.device, dtype=torch.float32) * self.dx

        # phi_focus(r, lambda) = -(2*pi/lambda) * (sqrt(r^2 + f^2) - f)
        phi_focus = -(2 * torch.pi / lambda_ref) * (torch.sqrt(r * r + self.f * self.f) - self.f)
        # wrap to [0, 2pi)
        phi_focus = torch.remainder(phi_focus, 2 * torch.pi)

        # interpolate radius for target focusing phase
        init_r = self.differentiable_interp1d(
            self.phase_broadband[ref_idx],
            self.radius_values.squeeze(),            
            phi_focus,
        )

        # clamp and convert to logits for your parameterization
        init_r = init_r.clamp(self.radius_min + 1e-12, self.radius_max - 1e-12)
        p = (init_r - self.radius_min) / (self.radius_max - self.radius_min)
        logits0 = torch.logit(p) / self.SCALE

        self.radius_vector_learnable = torch.nn.Parameter(logits0, requires_grad=True)
        self.radius_map_init = init_r
        

    @staticmethod
    def differentiable_interp1d(x_table, y_table, x_query):
        x_table = x_table.to(x_query.device)
        y_table = y_table.to(x_query.device)

        sorted_idx = torch.argsort(x_table)
        x_table = x_table[sorted_idx]
        y_table = y_table[sorted_idx]

        idx_upper = torch.searchsorted(x_table, x_query, right=True)
        idx_upper = torch.clamp(idx_upper, 1, x_table.shape[0] - 1)
        idx_lower = idx_upper - 1

        x0 = x_table[idx_lower]
        x1 = x_table[idx_upper]
        y0 = y_table[idx_lower]
        y1 = y_table[idx_upper]

        t = (x_query - x0) / (x1 - x0 + 1e-12)
        return y0 + t * (y1 - y0)

    def _z_list(self):
        if self.n_z == 1:
            return [self.f]
        z0 = self.f - 0.5 * self.z_span
        z1 = self.f + 0.5 * self.z_span
        zs = torch.linspace(z0, z1, steps=self.n_z, device=self.device, dtype=torch.float32)
        return [float(z) for z in zs]

    def forward(self):
        radius_vector = self.radius_min + (self.radius_max - self.radius_min) * torch.sigmoid(self.SCALE * self.radius_vector_learnable)
        radius_map = rotate_radius_vector_nearest(radius_vector, self.N, center=None)

        z_list = self._z_list()

        psf_list = []
        phase_map_list = []
        for c in range(self.n_lambda):
            lambda_m = float(self.wavelengths[c].item())

            phase_map = self.differentiable_interp1d(
                self.radius_values.squeeze(),
                self.phase_broadband[c],
                radius_map,
            )

            u0 = self.aperture_mask * torch.exp(1j * phase_map.to(torch.complex64))

            psf_depth = []
            for z in z_list:
                uz = propagate_angularspec_fast(
                    u0,
                    M=self.N,
                    lambda_m=lambda_m,
                    dx=self.dx,
                    z=float(z),
                    use_gpu=True,
                    bandlimit=True,
                )
                I = (uz.abs() ** 2).to(torch.float32)
                I = I / (I.sum() + 1e-12)
                psf_depth.append(I[None, None, :, :])

            psf_c = torch.cat(psf_depth, dim=0)
            psf_list.append(psf_c)
            phase_map_list.append(phase_map)

        return psf_list, radius_map, radius_vector, self.aperture_mask, phase_map_list, self.radius_map_init


#######################################################################
####DUMMY TEST CODE TO GENERATE PSF AND RADIUS MAP VISUALIZATION#####
#######################################################################

if __name__ == "__main__":
    import os
    import torch
    import matplotlib.pyplot as plt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mat_files_directory = r"C:\Users\kranat\Desktop\JesseMTF_optimization\meta_data"
    out_dir = "./psf_debug_outputs"
    os.makedirs(out_dir, exist_ok=True)

    wavelengths_m = [681e-9, 601e-9, 521e-9, 433e-9]

    N = 5000 #to reduces boundary artifacts in the PSF, use a larger grid size than the metalens diameter. The metalens diameter is 0.5 mm, so with dx=500 nm, we need at least 1000 pixels to cover the metalens, and we add some margin for better PSF quality.
    dx_m = 225e-9 #metalens sampling interval in meters per pixel, also the pixel size of the radius map. 
    focal_length_m = 8e-3
    metalens_diameter_m = 1e-3

    optic = OpticsModel(
        mat_files_directory=mat_files_directory,
        wavelengths_m=wavelengths_m,
        N=N,
        dx_m=dx_m,
        focal_length_m=focal_length_m,
        metalens_diameter_m=metalens_diameter_m,
        n_z=1,
        z_span_m=0.0,
        device=device,
    ).to(device)

    optic.eval()
    with torch.no_grad():
        psf_list, radius_map, radius_vector, aperture_mask, phase_map, radius_map_init = optic()

    print("radius_map nm min max:", float(radius_map.min() * 1e9), float(radius_map.max() * 1e9))
    print("radius_vector nm min max:", float(radius_vector.min() * 1e9), float(radius_vector.max() * 1e9))

    radius_map_plot = radius_map * aperture_mask

    for i, psf in enumerate(psf_list):
        psf2d = psf[0, 0].detach().float().cpu().numpy()
        s = psf2d.sum()
        mx = psf2d.max()

        lam_nm = wavelengths_m[i] * 1e9
        print(f"lambda {lam_nm:.0f} nm  psf shape {psf.shape}  sum {s:.6f}  max {mx:.6e}")

        plt.figure()
        plt.imshow(psf2d, cmap="gray")
        plt.title(f"PSF  {lam_nm:.0f} nm  sum={s:.4f}")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"psf_{lam_nm:.0f}nm.png"), dpi=200)
        plt.close()

        H, W = psf2d.shape
        cy = H // 2
        cx = W // 2

        plt.figure()
        plt.plot(psf2d[cy, :])
        plt.title(f"Center row  {lam_nm:.0f} nm")
        plt.xlabel("pixel")
        plt.ylabel("intensity")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"psf_profile_row_{lam_nm:.0f}nm.png"), dpi=200)
        plt.close()

        plt.figure()
        plt.plot(psf2d[:, cx])
        plt.title(f"Center col  {lam_nm:.0f} nm")
        plt.xlabel("pixel")
        plt.ylabel("intensity")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"psf_profile_col_{lam_nm:.0f}nm.png"), dpi=200)
        plt.close()

    rm = radius_map_plot.detach().float().cpu().numpy()
    plt.figure()
    plt.imshow(rm * 1e9, cmap="viridis")
    plt.title("Radius map (nm)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "radius_map_nm.png"), dpi=200)
    plt.close()

    # -----------------------------
    # Plot phase map per wavelength (phase_map is a list of tensors)
    # -----------------------------
    def wrap_to_pi(x):
        return ((x + torch.pi) % (2 * torch.pi)) - torch.pi

    phase_out_dir = os.path.join(out_dir, "phase_maps")
    os.makedirs(phase_out_dir, exist_ok=True)

    for i, ph in enumerate(phase_map):
        lam_nm = wavelengths_m[i] * 1e9

        ph2d = ph.detach().float()
        ph2d = ph2d * aperture_mask  # optional, masks outside metalens
        ph2d = wrap_to_pi(ph2d)      # optional, wrap for nicer colormap

        ph_np = ph2d.cpu().numpy()

        print(
            f"lambda {lam_nm:.0f} nm  phase shape {tuple(ph2d.shape)}  "
            f"min {float(ph2d.min()):.4f}  max {float(ph2d.max()):.4f}"
        )

        plt.figure()
        plt.imshow(ph_np, cmap="twilight", vmin=-torch.pi, vmax=torch.pi)
        plt.title(f"Phase map  {lam_nm:.0f} nm")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(phase_out_dir, f"phase_{lam_nm:.0f}nm.png"), dpi=200)
        plt.close()


    print("Saved outputs to:", os.path.abspath(out_dir))
