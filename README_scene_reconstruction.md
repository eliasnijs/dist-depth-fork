# Scene Reconstruction from Image Sequence

This utility combines multiple images (typically video frames) into a complete 3D scene reconstruction. It uses depth estimation, point cloud generation, and cloud registration to create a coherent 3D model of the scene.

## Requirements

The script requires the following dependencies (added to pyproject.toml):
- open3d
- tqdm
- torch
- opencv-python
- numpy
- matplotlib

## Basic Usage

Process a sequence of images to create a 3D scene:

```bash
uv run python scene_reconstruction.py --folder path/to/image/folder --output scene_results --save_depth
```

## Required Arguments

- `--folder` : Path to the folder containing image sequence

## Optional Arguments

### Output Options
- `--output` : Output directory (default: "scene_output")
- `--save_depth` : Save depth maps as PNG and NPY files
- `--save_individual_pcs` : Save individual point clouds before merging

### Camera Parameters
- `--focal_length` : Focal length of the camera (default: 886.81)
- `--center_x` : Principal point x-coordinate (default: image width/2)
- `--center_y` : Principal point y-coordinate (default: image height/2)
- `--width` : Camera width (default: detected from image)
- `--height` : Camera height (default: detected from image)
- `--force_camera_params` : Force specified camera parameters instead of using image dimensions

### Depth Estimation
- `--min_depth` : Minimum depth value (default: 0.1)
- `--max_depth` : Maximum depth value (default: 10.0)
- `--input_size` : Size for neural network input (default: 256)
- `--output_size` : Size for neural network output (default: 512)
- `--ckpt_path` : Path to the checkpoints directory (default: ./ckpts)

### Registration & Reconstruction Parameters
- `--voxel_size` : Voxel size for downsampling point clouds (default: 0.01)
- `--icp_distance_threshold` : Distance threshold for ICP registration (default: 0.05)
- `--keyframe_interval` : Interval for selecting keyframes (default: 5)
- `--overlap_threshold` : Overlap threshold for point cloud registration (default: 0.7)
- `--global_registration` : Use global registration with RANSAC (slower but more robust)

### System
- `--no_cuda` : If set, disables CUDA even if available
- `--image_ext` : Comma-separated list of image extensions to process (default: "jpg,jpeg,png")

## How It Works

The scene reconstruction process involves several steps:

1. **Depth Estimation**: Each frame is processed to generate a depth map using a pre-trained neural network
2. **Point Cloud Generation**: RGB-D information is converted to 3D point clouds
3. **Camera Pose Estimation**: The script tries to use:
   - Pose estimation networks if available (pose_encoder.pth and pose.pth)
   - OR point cloud registration techniques (ICP and global registration)
4. **Point Cloud Registration**: Individual point clouds are aligned and transformed to a common coordinate system
5. **Scene Integration**: Registered point clouds are combined, downsampled, and filtered
6. **Mesh Generation**: The script attempts to create a textured mesh using Poisson reconstruction

## Tips for Better Results

1. **Use Sequential Frames**: Images should be from a video or sequential captures of the same scene
2. **Sufficient Overlap**: Ensure adjacent frames have enough overlap (>60%)
3. **Adjust Key Parameters**:
   - `--keyframe_interval`: Lower values (1-3) give better registration but slower processing
   - `--voxel_size`: Smaller values give more detail but slower processing
   - `--global_registration`: Enable for challenging scenes
4. **Camera Parameters**: For best results, provide accurate camera intrinsics
5. **Sufficient Lighting**: Well-lit scenes produce better depth estimation and registration

## Output Files

The script generates:
- `scene_reconstruction.ply`: The combined point cloud of the entire scene
- `scene_mesh.obj`: A textured mesh (if mesh generation was successful)
- Individual depth maps and point clouds (if requested with `--save_depth` and `--save_individual_pcs`)

## Viewing Results

The output PLY and OBJ files can be viewed in any 3D viewer that supports these formats, such as:
- MeshLab
- CloudCompare
- Blender