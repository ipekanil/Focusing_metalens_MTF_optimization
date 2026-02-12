# Multispectral focusing metalens optimization 

This repository optimizes a rotationally symmetric metalens design for multiple wavelengths using only physics based PSF simulation and frequency domain losses. The design variable is a 1D radial radius vector that is expanded into a 2D radius map, converted to wavelength dependent phase using a precomputed TiO2 library, and propagated to the image plane with the angular spectrum method.

## What it does

- Builds a learnable 1D radius profile (in meters) within fabrication limits.

- Expands it to a 2D rotationally symmetric radius map.

- Converts radius map to phase map for each wavelength using the library lookup.

- Forms the complex field at the metalens plane (aperture times exp(i phase)).

- Propagates to the focal plane (or multiple z planes if enabled) using angular spectrum propagation.

- Computes PSFs for all wavelengths.

- Optimizes the radius vector using a combination of MTF shaping loss and focusing loss (encircled energy).

## Files

### OpticsModel.py
Physics forward model. Loads the radius and phase libraries from meta_data, creates the aperture mask, initializes the radius vector using a focusing phase at a reference wavelength, then returns PSFs and intermediate maps. 

### OpticsOnlyLit.py
PyTorch Lightning module that runs optics only training with a dummy dataloader. Computes losses from the PSFs and updates radius_vector_learnable. Logs loss terms and radius update statistics. 

### main.py
Training entry point. Defines a Config, sets simulation parameters, starts Lightning trainer, and saves checkpoints. 

### visualize_trained_optics.py
Loads a trained checkpoint and saves visualizations for PSF, MTF, phase maps, and radius maps (initial, final, and difference). 

### meta_data/
Must contain the library files:

edge_PRGB_TiO2_225P_1um_h.mat with edge (radius values)

phase_PRGB_TiO2_225P_1um_h.mat with phase_matrix (phase versus radius for each wavelength)

## Installation

This code uses PyTorch, PyTorch Lightning, NumPy, h5py, matplotlib, and optionally Weights and Biases.

### A minimal pip install list:
```bash
pip install torch pytorch-lightning numpy h5py matplotlib
pip install wandb
```
# Usage
## Quick start
### 1) Train

Edit parameters inside Config in 'main.py' if needed, then run:

```bash
python main.py
```
Checkpoints are saved to:
```bash
checkpoints_optics_only/ 
```

### 2) Visualize a trained model

Update ckpt_path inside 'visualize_trained_optics.py' to the checkpoint you want, then run:

```bash
python visualize_trained_optics.py
```
Outputs are saved to:
```bash
viz_outputs/ 
```
## Key parameters

**Simulation grid**

N is the simulation grid size in pixels.

dx_m is the sampling at the metalens plane in meters per pixel.

The full simulated window size is L = N * dx_m.

The metalens diameter must satisfy:
metalens_diameter_m <= N * dx_m
This is checked in code and will raise an error if violated.

Practical note: choose N larger than metalens_diameter_m / dx_m to reduce boundary artifacts in the PSF.

**Wavelengths**

wavelengths_m is a list of wavelengths in meters, for example:
[681e-9, 601e-9, 521e-9, 433e-9]

**Focal length and z planes**

focal_length_m is the target focus distance.

If n_z == 1, the model propagates only to z = focal_length_m.

If n_z > 1, it simulates multiple planes spanning z_span_m centered at the focal length. 

## Library

### **OpticsModel**

For each wavelength:

Radius map is converted to phase map using a differentiable 1D interpolation of the library table. 

Complex field at the metalens plane is:
u0 = aperture_mask * exp(i * phase_map)

Propagation uses angular spectrum:
u(z) = IFFT( FFT(u0) * H(fx, fy, z) )
with optional band limiting to suppress evanescent components. 

PSF is |u(z)|^2 normalized to sum to 1.

### **Optimization variables**

The learned parameter is radius_vector_learnable which is stored in logit space for stable constraints. Physical radii are produced by:

radius_vector = radius_min + (radius_max - radius_min) * sigmoid(SCALE * radius_vector_learnable)

The 2D rotationally symmetric map is created by nearest ring assignment.

## **Initialization**

The default initialization builds a focusing phase profile at one reference wavelength and inverts the library to obtain an initial radius vector. This gives a reasonable starting point that focuses well at the reference wavelength. 

**OpticsModel**

You can change the reference wavelength using ref_idx inside OpticsModel.__init__. 

## **Losses**

Training uses two main objectives: 

**MTF shaping loss (edof_mtf_rmse_simple)**
Computes the MTF from the PSF and compares its radial profile to a target curve up to rho_cut. Aggregation uses a worst wavelength soft max to push the weakest channel.

**Focusing loss (encircled_energy_loss)**
Maximizes encircled energy inside a small radius around the PSF center. In code it is implemented as 1 - EE, and uses worst channel reduction to enforce co focusing.

**Total loss:**
loss = w_mtf * mtf_loss + w_focus * focus_loss

## **Logging and tracking radius updates**

During training, the Lightning module logs:

- train/radius_grad_norm

- train/radius_delta_mean_nm

- train/radius_delta_max_nm

- min and max radius values

These are useful to verify the parameter is updating and not stuck. 

# **Common tips**

If only the reference wavelength focuses well, increase pressure on the worst wavelength by keeping reduce_channel="max" in encircled_energy_loss and using the worst channel aggregation in the MTF loss.

If the radius updates are tiny, increase lr_optical or reduce SCALE. If the loss is unstable, reduce lr_optical.

If PSFs show ringing or boundary artifacts, increase N while keeping metalens_diameter_m fixed.

# **Outputs**

The visualization script saves:

- PSF images for each wavelength and z index

- MTF images for each PSF

- Phase maps per wavelength

- Radius map final, radius map init, and their difference 

## License

MIT License. See `LICENSE`.

## Citation

Please cite this repository if you use it. A CITATION.cff file is included.
