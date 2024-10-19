# Jittor version of "3D Gaussian Splatting"
This repository contains the implementation with jittor for paper "3D Gaussian Splatting for Real-Time Radiance Field Rendering". In our implementation, the speed of evaluation script is **10x** faster than PyTorch version.

## Set-up
### Requirements
```
jittor
cmake
CUDA>=11
g++
plyfile
tqdm
```
### Compile the submodules
The simple-knn and diff_gaussian_rasterizater modules should be compiled with cmake and make:
```
cd gaussian-renderer/diff_gaussian_rasterizater
cmake .
make -j
cd ../../scene/simple-knn
cmake .
make -j
```
You will get simpleknn.so and CudaRasterizer.so in simple-knn and diff_gaussian_rasterizater folders. 
### LPIPS
The repository uses [Jittor_Perceptual-Similarity-Metric](https://github.com/ty625911724/Jittor_Perceptual-Similarity-Metric) for evaluation. Please download the pretrained model following the origin repository
 and put the weight file in lpips_jittor folder. 
 
## Running
To run the optimizer, simply use:
```
python train.py -s <path to COLMAP or NeRF Synthetic dataset>
```
<details>
<summary><span style="font-weight: bold;">Command Line Arguments for train.py</span></summary>

  #### --source_path / -s
  Path to the source directory containing a COLMAP or Synthetic NeRF data set.
  #### --model_path / -m 
  Path where the trained model should be stored (```output/<random>``` by default).
  #### --images / -i
  Alternative subdirectory for COLMAP images (```images``` by default).
  #### --eval
  Add this flag to use a MipNeRF360-style training/test split for evaluation.
  #### --resolution / -r
  Specifies resolution of the loaded images before training. If provided ```1, 2, 4``` or ```8```, uses original, 1/2, 1/4 or 1/8 resolution, respectively. For all other values, rescales the width to the given number while maintaining image aspect. **If not set and input image width exceeds 1.6K pixels, inputs are automatically rescaled to this target.**
  #### --data_device
  Specifies where to put the source image data, ```cuda``` by default, recommended to use ```cpu``` if training on large/high-resolution dataset, will reduce VRAM consumption, but slightly slow down training. Thanks to [HrsPythonix](https://github.com/HrsPythonix).
  #### --white_background / -w
  Add this flag to use white background instead of black (default), e.g., for evaluation of NeRF Synthetic dataset.
  #### --sh_degree
  Order of spherical harmonics to be used (no larger than 3). ```3``` by default.
  #### --convert_SHs_python
  Flag to make pipeline compute forward and backward of SHs with PyTorch instead of ours.
  #### --convert_cov3D_python
  Flag to make pipeline compute forward and backward of the 3D covariance with PyTorch instead of ours.
  #### --debug
  Enables debug mode if you experience erros. If the rasterizer fails, a ```dump``` file is created that you may forward to us in an issue so we can take a look.
  #### --debug_from
  Debugging is **slow**. You may specify an iteration (starting from 0) after which the above debugging becomes active.
  #### --iterations
  Number of total iterations to train for, ```30_000``` by default.
  #### --ip
  IP to start GUI server on, ```127.0.0.1``` by default.
  #### --port 
  Port to use for GUI server, ```6009``` by default.
  #### --test_iterations
  Space-separated iterations at which the training script computes L1 and PSNR over test set, ```7000 30000``` by default.
  #### --save_iterations
  Space-separated iterations at which the training script saves the Gaussian model, ```7000 30000 <iterations>``` by default.
  #### --checkpoint_iterations
  Space-separated iterations at which to store a checkpoint for continuing later, saved in the model directory.
  #### --start_checkpoint
  Path to a saved checkpoint to continue training from.
  #### --quiet 
  Flag to omit any text written to standard out pipe. 
  #### --feature_lr
  Spherical harmonics features learning rate, ```0.0025``` by default.
  #### --opacity_lr
  Opacity learning rate, ```0.05``` by default.
  #### --scaling_lr
  Scaling learning rate, ```0.005``` by default.
  #### --rotation_lr
  Rotation learning rate, ```0.001``` by default.
  #### --position_lr_max_steps
  Number of steps (from 0) where position learning rate goes from ```initial``` to ```final```. ```30_000``` by default.
  #### --position_lr_init
  Initial 3D position learning rate, ```0.00016``` by default.
  #### --position_lr_final
  Final 3D position learning rate, ```0.0000016``` by default.
  #### --position_lr_delay_mult
  Position learning rate multiplier (cf. Plenoxels), ```0.01``` by default. 
  #### --densify_from_iter
  Iteration where densification starts, ```500``` by default. 
  #### --densify_until_iter
  Iteration where densification stops, ```15_000``` by default.
  #### --densify_grad_threshold
  Limit that decides if points should be densified based on 2D position gradient, ```0.0002``` by default.
  #### --densification_interval
  How frequently to densify, ```100``` (every 100 iterations) by default.
  #### --opacity_reset_interval
  How frequently to reset opacity, ```3_000``` by default. 
  #### --lambda_dssim
  Influence of SSIM on total loss from 0 to 1, ```0.2``` by default. 
  #### --percent_dense
  Percentage of scene extent (0--1) a point must exceed to be forcibly densified, ```0.01``` by default.

</details>
<br>

### Evaluation

By default, the trained models use all available images in the dataset. To train them while withholding a test set for evaluation, use the ```--eval``` flag. This way, you can render training/test sets and produce error metrics as follows:
```shell
python train.py -s <path to COLMAP or NeRF Synthetic dataset> --eval # Train with train/test split
python render.py -m <path to trained model> # Generate renderings
python metrics.py -m <path to trained model> # Compute error metrics on renderings
```

## Plan of Models
JGaussian will support more valuable 3DGS models in the future, if you are also interested in JGaussian and want to improve it, welcome to submit PR!  
<b>:heavy_check_mark:Supported  :clock3:Doing :heavy_plus_sign:TODO</b>
- :heavy_check_mark: 3D Gaussian Splatting
- :heavy_check_mark: [Mip-Splatting](https://github.com/lishaobingdong/mip-splatting-jittor)
- :clock3: FSGS: Real-Time Few-Shot View Synthesis using Gaussian Splatting
- :clock3: 2D Gaussian Splatting for Geometrically Accurate Radiance Fields
- :heavy_plus_sign: PGSR: Planar-based Gaussian Splatting for Efficient and High-Fidelity Surface Reconstruction
- :heavy_plus_sign: EVSplitting: An Efficient and Visually Consistent Splitting Algorithm for 3D Gaussian Splatting
- :heavy_plus_sign: ...
## Acknowledgements
The original implementation comes from the following cool project:
* [3DGS](https://github.com/graphdeco-inria/gaussian-splatting/)
* [Jittor-LPIPS](https://github.com/ty625911724/Jittor_Perceptual-Similarity-Metric)
