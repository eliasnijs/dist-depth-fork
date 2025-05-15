# Image to Point Cloud Conversion Utilities

This repository provides command-line utilities for converting RGB images to 3D point clouds.

## Installation

These utilities are part of the dist-depth-fork repository and require the dependencies specified in `pyproject.toml`. It's recommended to use `uv` to handle dependencies.

```bash
# Set up environment with UV
uv run python image_to_pc.py --help
uv run python folder_to_pc.py --help
```

## Single Image Conversion (image_to_pc.py)

Convert a single image to a point cloud.

### Basic Usage

```bash
uv run python image_to_pc.py --image path/to/image.jpg --output output_directory --save_depth
```

### Required Arguments

- `--image` : Path to the input RGB image

### Optional Arguments

- `--output` : Output directory (default: creates 'pc_output' in the current directory)
- `--save_depth` : Save depth map as PNG and NPY files
- `--ckpt_path` : Path to the checkpoints directory containing encoder.pth and depth.pth (default: ./ckpts)
- `--no_cuda` : If set, disables CUDA even if available

## Batch Folder Conversion (folder_to_pc.py)

Process all images in a folder and convert them to point clouds. Uses efficient batch processing to process multiple images in parallel, significantly speeding up processing when using a GPU.

### Basic Usage

```bash
uv run python folder_to_pc.py --folder path/to/image/folder --output output_directory --save_depth
```

### Required Arguments

- `--folder` : Path to the folder containing input RGB images

### Optional Arguments

- `--output` : Output directory (default: creates 'pc_output' in the current directory)
- `--save_depth` : Save depth map as PNG and NPY files
- `--ckpt_path` : Path to the checkpoints directory containing encoder.pth and depth.pth (default: ./ckpts)
- `--no_cuda` : If set, disables CUDA even if available
- `--image_ext` : Comma-separated list of image extensions to process (default: jpg,jpeg,png)
- `--skip_existing` : Skip processing images that already have corresponding point clouds
- `--force_camera_params` : Force using the specified camera parameters instead of image dimensions

### Batch Processing Options

- `--batch_size` : Number of images to process in parallel (default: auto-determined based on device)
- `--no_batch` : Disable batch processing and process images one by one

## Camera Parameters

The following parameters can be customized to adjust the camera model for point cloud generation:

- `--focal_length` : Focal length of the camera (default: 886.81)
- `--center_x` : Principal point x-coordinate (default: image width/2)
- `--center_y` : Principal point y-coordinate (default: image height/2)
- `--width` : Camera width (default: detected from image)
- `--height` : Camera height (default: detected from image)

## Depth Estimation Parameters

- `--min_depth` : Minimum depth value (default: 0.1)
- `--max_depth` : Maximum depth value (default: 10.0)
- `--input_size` : Size for neural network input (default: 256)
- `--output_size` : Size for neural network output (default: 512)

## Examples

Convert a single image to a point cloud:

```bash
uv run python image_to_pc.py --image data/sample_pc/0001.jpg --output results
```

Process all images in a folder:

```bash
uv run python folder_to_pc.py --folder data/sample_pc --output results --save_depth
```

Skip already processed images in a folder:

```bash
uv run python folder_to_pc.py --folder data/sample_pc --output results --save_depth --skip_existing
```

Set specific camera parameters:

```bash
uv run python folder_to_pc.py --folder data/sample_pc --output results --focal_length 500 --center_x 256 --center_y 256 --force_camera_params
```

Process a large folder with a specific batch size (for systems with a lot of GPU memory):

```bash
uv run python folder_to_pc.py --folder data/large_dataset --output results --batch_size 8
```

Disable batch processing for lower memory usage:

```bash
uv run python folder_to_pc.py --folder data/sample_pc --output results --no_batch
```

## Output Files

For each input image `example.jpg`, the following files will be created:

- `example_pc.ply` : The 3D point cloud in PLY format
- `example_depth.npy` : The depth map as a NumPy array (if --save_depth is used)
- `example_depth.png` : Visualization of the depth map (if --save_depth is used)

## Viewing the Results

You can view the PLY files using any 3D viewer that supports the PLY format, such as:

- MeshLab
- CloudCompare
- Blender

## Notes

- The tools automatically adjust camera parameters based on the input image dimensions unless forced with --force_camera_params
- Files with "_depth" in their name are automatically filtered out when processing folders
- For best results, use images with clear depth cues
- The depth estimation model is pre-trained and works best with indoor scenes