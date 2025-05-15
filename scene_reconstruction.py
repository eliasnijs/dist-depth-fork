#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import cv2
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import os
import torch
import glob
import open3d as o3d
from tqdm import tqdm
from networks.depth_decoder import DepthDecoder
from networks.pose_decoder import PoseDecoder
from networks.resnet_encoder import ResnetEncoder
from utils import output_to_depth
from visualize_pc import generate_pointcloud

def parse_args():
    parser = argparse.ArgumentParser(description="Scene Reconstruction from Image Sequence")
    
    parser.add_argument("--folder", type=str, required=True,
                        help="Path to the input image folder (video frames)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: creates 'scene_output' in the current directory)")
    parser.add_argument("--ckpt_path", type=str, default="./ckpts",
                        help="Path to the checkpoints directory containing network weights")
    parser.add_argument("--focal_length", type=float, default=886.81,
                        help="Focal length of the camera")
    parser.add_argument("--center_x", type=float, default=512.0, 
                        help="Principal point x-coordinate")
    parser.add_argument("--center_y", type=float, default=384.0,
                        help="Principal point y-coordinate")
    parser.add_argument("--width", type=int, default=1024,
                        help="Camera width")
    parser.add_argument("--height", type=int, default=768, 
                        help="Camera height")
    parser.add_argument("--force_camera_params", action="store_true",
                        help="Force using the specified camera parameters instead of image dimensions")
    parser.add_argument("--min_depth", type=float, default=0.1,
                        help="Minimum depth value")
    parser.add_argument("--max_depth", type=float, default=10.0,
                        help="Maximum depth value")
    parser.add_argument("--save_depth", action="store_true",
                        help="Save depth maps as PNG and NPY files")
    parser.add_argument("--save_individual_pcs", action="store_true",
                        help="Save individual point clouds before merging")
    parser.add_argument("--input_size", type=int, default=256,
                        help="Size for neural network input (default: 256)")
    parser.add_argument("--output_size", type=int, default=512,
                        help="Size for neural network output (default: 512)")
    parser.add_argument("--no_cuda", action="store_true",
                        help="If set, disables CUDA even if available")
    parser.add_argument("--image_ext", type=str, default="jpg,jpeg,png",
                        help="Comma-separated list of image extensions to process (default: jpg,jpeg,png)")
    parser.add_argument("--voxel_size", type=float, default=0.01,
                        help="Voxel size for downsampling point clouds (default: 0.01)")
    parser.add_argument("--icp_distance_threshold", type=float, default=0.05,
                        help="Distance threshold for ICP registration (default: 0.05)")
    parser.add_argument("--keyframe_interval", type=int, default=5,
                        help="Interval for selecting keyframes (default: 5)")
    parser.add_argument("--overlap_threshold", type=float, default=0.7,
                        help="Overlap threshold for point cloud registration (default: 0.7)")
    parser.add_argument("--global_registration", action="store_true",
                        help="Use global registration with RANSAC (slower but more robust)")
    
    return parser.parse_args()

def setup_camera_params(args, width, height):
    """Set global camera parameters for point cloud generation"""
    import visualize_pc
    visualize_pc.focalLength = args.focal_length
    visualize_pc.centerX = args.center_x if args.force_camera_params else width / 2
    visualize_pc.centerY = args.center_y if args.force_camera_params else height / 2
    visualize_pc.intWidth = args.width if args.force_camera_params else width
    visualize_pc.intHeight = args.height if args.force_camera_params else height

def get_ordered_image_files(folder_path, extensions):
    """Get list of image files with specified extensions, sorted by name (assumed to be frames)"""
    image_files = []
    ext_list = extensions.lower().split(',')
    
    for ext in ext_list:
        ext = ext.strip()
        pattern = os.path.join(folder_path, f"*.{ext}")
        image_files.extend(glob.glob(pattern))
        # Also check for uppercase extensions
        pattern = os.path.join(folder_path, f"*.{ext.upper()}")
        image_files.extend(glob.glob(pattern))
    
    # Filter out depth images by filename pattern
    filtered_files = []
    for file in image_files:
        # Skip files with "_depth" in the filename
        if "_depth" not in os.path.basename(file):
            filtered_files.append(file)
    
    return sorted(filtered_files)

def process_image(image_path, device, encoder, depth_decoder, input_size, output_size, min_depth, max_depth):
    """Process an image and return the depth map"""
    # Read image
    img = cv2.imread(image_path, -1)
    if img is None:
        return None, None, None, None
        
    # Get original dimensions for later resizing
    original_height, original_width = img.shape[:2]
    
    # Process for network
    raw_img = np.transpose(img[:, :, :3], (2, 0, 1))
    
    # Convert to tensor
    input_image = torch.from_numpy(raw_img).float().to(device)
    input_image = (input_image / 255.0).unsqueeze(0)
    
    # Resize for network input
    input_image = torch.nn.functional.interpolate(
        input_image, (input_size, input_size), mode="bilinear", align_corners=False
    )
    
    # Forward pass
    features = encoder(input_image)
    outputs = depth_decoder(features)
    
    # Process output
    out = outputs[("out", 0)]
    
    # Resize to original image dimensions to ensure compatibility with point cloud generation
    out_resized = torch.nn.functional.interpolate(
        out, (original_height, original_width), mode="bilinear", align_corners=False
    )
    
    # Convert to depth
    depth = output_to_depth(out_resized, min_depth, max_depth)
    metric_depth = depth.detach().cpu().numpy().squeeze()
    
    return metric_depth, original_width, original_height, img

def estimate_pose(prev_image, curr_image, device, pose_encoder, pose_decoder):
    """Estimate camera transformation from two consecutive frames"""
    # This is a simplified version assuming we have a pose network
    # In practice, you might need SfM or visual odometry
    
    # Preprocess images to tensor format
    prev_tensor = torch.from_numpy(np.transpose(prev_image, (2, 0, 1))).float().to(device).unsqueeze(0) / 255.0
    curr_tensor = torch.from_numpy(np.transpose(curr_image, (2, 0, 1))).float().to(device).unsqueeze(0) / 255.0
    
    # Concatenate images for pose network
    pose_input = torch.cat([prev_tensor, curr_tensor], 1)
    
    # Resize for network
    pose_input = torch.nn.functional.interpolate(pose_input, (256, 256), mode="bilinear", align_corners=False)
    
    # Get pose features and prediction
    pose_features = pose_encoder(pose_input)
    pose = pose_decoder([pose_features])
    
    # Extract 6-DoF pose parameters (3 for translation, 3 for rotation)
    axisangle = pose[0, :3].cpu().detach().numpy()
    translation = pose[0, 3:].cpu().detach().numpy()
    
    # Convert axis-angle to rotation matrix
    rod_rot = cv2.Rodrigues(axisangle)[0]
    
    # Create 4x4 transformation matrix
    transformation = np.eye(4)
    transformation[:3, :3] = rod_rot
    transformation[:3, 3] = translation
    
    return transformation

def generate_point_cloud(rgb_image, depth_map, intrinsic, filename=None):
    """Generate point cloud from RGB and depth images"""
    # Create Open3D RGBD image
    rgb_o3d = o3d.geometry.Image(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
    depth_o3d = o3d.geometry.Image(depth_map.astype(np.float32))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_o3d, depth_o3d, 
        depth_scale=1.0, depth_trunc=10.0, convert_rgb_to_intensity=False
    )
    
    # Create point cloud
    intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(
        width=depth_map.shape[1],
        height=depth_map.shape[0],
        fx=intrinsic[0, 0],
        fy=intrinsic[1, 1],
        cx=intrinsic[0, 2],
        cy=intrinsic[1, 2]
    )
    
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic_o3d)
    
    # Save point cloud if filename is provided
    if filename:
        o3d.io.write_point_cloud(filename, pcd)
    
    return pcd

def preprocess_point_cloud(pcd, voxel_size):
    """Preprocess point cloud for registration"""
    # Downsample
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    # Estimate normals
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    
    # Compute FPFH features for global registration
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
    
    return pcd_down, pcd_fpfh

def register_point_clouds(source, target, source_fpfh, target_fpfh, voxel_size, use_global=True):
    """Register two point clouds"""
    # Initial alignment using global registration
    if use_global:
        result_global = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source, target, source_fpfh, target_fpfh, True,
            voxel_size * 1.5,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 1.5)
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
        initial_transform = result_global.transformation
    else:
        # Identity transform if not using global registration
        initial_transform = np.identity(4)
    
    # Refine alignment using ICP
    result_icp = o3d.pipelines.registration.registration_icp(
        source, target, voxel_size, initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
    
    return result_icp.transformation

def main():
    args = parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Check if folder exists
    if not os.path.isdir(args.folder):
        print(f"Error: {args.folder} is not a valid directory")
        return
    
    # Create output directory if needed
    output_dir = args.output if args.output else "scene_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create directories for intermediate results if needed
    depth_dir = os.path.join(output_dir, "depth")
    pc_dir = os.path.join(output_dir, "point_clouds")
    
    if args.save_depth and not os.path.exists(depth_dir):
        os.makedirs(depth_dir)
    if args.save_individual_pcs and not os.path.exists(pc_dir):
        os.makedirs(pc_dir)
    
    # Get ordered image files
    image_files = get_ordered_image_files(args.folder, args.image_ext)
    if not image_files:
        print(f"No images with extensions {args.image_ext} found in {args.folder}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Load neural networks
    print("Loading neural networks...")
    with torch.no_grad():
        # Load depth estimation networks
        encoder = ResnetEncoder(152, False)
        depth_decoder = DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))
        
        # Initialize for pose networks if needed
        pose_encoder = None
        pose_decoder = None
        has_pose_network = False
        
        # Load depth estimation weights (required)
        try:
            # Load depth encoder
            encoder_path = os.path.join(args.ckpt_path, "encoder.pth")
            loaded_dict_enc = torch.load(encoder_path, map_location=device)
            filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
            encoder.load_state_dict(filtered_dict_enc)
            encoder.to(device)
            encoder.eval()
            
            # Load depth decoder
            depth_path = os.path.join(args.ckpt_path, "depth.pth")
            loaded_dict = torch.load(depth_path, map_location=device)
            depth_decoder.load_state_dict(loaded_dict)
            depth_decoder.to(device)
            depth_decoder.eval()
            
            # Try to load pose networks (optional)
            pose_enc_path = os.path.join(args.ckpt_path, "pose_encoder.pth")
            pose_dec_path = os.path.join(args.ckpt_path, "pose.pth")
            
            if os.path.exists(pose_enc_path) and os.path.exists(pose_dec_path):
                pose_encoder = ResnetEncoder(18, True)
                pose_decoder = PoseDecoder(pose_encoder.num_ch_enc, 1)
                
                pose_enc_dict = torch.load(pose_enc_path, map_location=device)
                pose_encoder.load_state_dict(pose_enc_dict)
                pose_encoder.to(device)
                pose_encoder.eval()
                
                pose_dec_dict = torch.load(pose_dec_path, map_location=device)
                pose_decoder.load_state_dict(pose_dec_dict)
                pose_decoder.to(device)
                pose_decoder.eval()
                
                has_pose_network = True
                print("Pose estimation networks loaded successfully")
            else:
                print("Pose network weights not found, using point cloud registration instead")
                has_pose_network = False
            
        except Exception as e:
            print(f"Error loading networks: {e}")
            print("Please make sure the depth estimation networks (encoder.pth and depth.pth) are available")
            return
    
    # Process all frames
    point_clouds = []
    transformations = [np.eye(4)]  # Identity transform for the first frame
    previous_image = None
    
    print("Processing frames and generating point clouds...")
    
    for i, image_file in enumerate(tqdm(image_files)):
        base_filename = os.path.splitext(os.path.basename(image_file))[0]
        
        # Skip frames that aren't keyframes if not the first frame
        if i > 0 and i % args.keyframe_interval != 0:
            continue
        
        # Process image to get depth
        depth_map, width, height, rgb_image = process_image(
            image_file, device, encoder, depth_decoder,
            args.input_size, args.output_size, args.min_depth, args.max_depth
        )
        
        if depth_map is None:
            print(f"Error: Could not process image {image_file}")
            continue
        
        # Setup camera parameters based on first frame
        if i == 0:
            # Set camera intrinsics
            if args.force_camera_params:
                fx = args.focal_length
                fy = args.focal_length
                cx = args.center_x
                cy = args.center_y
            else:
                fx = args.focal_length
                fy = args.focal_length
                cx = width / 2
                cy = height / 2
                
                # Update args for visualization
                args.width = width
                args.height = height
                args.center_x = cx
                args.center_y = cy
            
            # Create camera intrinsic matrix
            intrinsic = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ])
            
            # Setup for point cloud generation
            setup_camera_params(args, width, height)
        
        # Save depth map
        if args.save_depth:
            np.save(os.path.join(depth_dir, f"{base_filename}_depth.npy"), depth_map)
            
            # Save depth visualization
            normalizer = mpl.colors.Normalize(vmin=args.min_depth, vmax=args.max_depth)
            mapper = cm.ScalarMappable(norm=normalizer, cmap="turbo")
            colormapped_im = (mapper.to_rgba(depth_map)[:, :, :3] * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(depth_dir, f"{base_filename}_depth.png"), colormapped_im[:,:,[2,1,0]])
        
        # Generate point cloud
        pc_path = os.path.join(pc_dir, f"{base_filename}_pc.ply") if args.save_individual_pcs else None
        pcd = generate_point_cloud(rgb_image, depth_map, intrinsic, pc_path)
        
        # Estimate pose if we have more than one frame and the pose network
        if i > 0 and has_pose_network and previous_image is not None:
            # Estimate transformation from previous frame to current frame
            try:
                transform = estimate_pose(previous_image, rgb_image, device, pose_encoder, pose_decoder)
                # Convert to global coordinate system by concatenating with previous transform
                global_transform = np.dot(transformations[-1], transform)
                transformations.append(global_transform)
            except Exception as e:
                print(f"Error estimating pose: {e}")
                # If pose estimation fails, use previous transform
                transformations.append(transformations[-1])
        elif i > 0:
            # No pose network, use point cloud registration instead
            try:
                # Downsample and compute features
                source_down, source_fpfh = preprocess_point_cloud(pcd, args.voxel_size)
                target_down, target_fpfh = preprocess_point_cloud(point_clouds[-1], args.voxel_size)
                
                # Register point clouds
                transform = register_point_clouds(
                    source_down, target_down, 
                    source_fpfh, target_fpfh, 
                    args.voxel_size, 
                    args.global_registration
                )
                
                # Convert to global coordinate system
                global_transform = np.dot(transformations[-1], transform)
                transformations.append(global_transform)
            except Exception as e:
                print(f"Error in point cloud registration: {e}")
                # If registration fails, use previous transform
                transformations.append(transformations[-1])
        
        # Apply transformation to point cloud
        if i > 0:
            pcd.transform(transformations[-1])
        
        # Add to list of point clouds
        point_clouds.append(pcd)
        
        # Store current image for next iteration
        previous_image = rgb_image
    
    # Combine all point clouds
    print("Combining point clouds...")
    combined_pcd = o3d.geometry.PointCloud()
    for pcd in point_clouds:
        combined_pcd += pcd
    
    # Down-sample combined point cloud to remove redundant points
    print("Down-sampling combined point cloud...")
    combined_pcd = combined_pcd.voxel_down_sample(args.voxel_size)
    
    # Apply statistical outlier removal
    print("Removing outliers...")
    combined_pcd, _ = combined_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    # Save combined point cloud
    output_path = os.path.join(output_dir, "scene_reconstruction.ply")
    o3d.io.write_point_cloud(output_path, combined_pcd)
    print(f"Scene reconstruction saved to {output_path}")
    
    # Create a textured mesh (optional, if reconstruction is good)
    try:
        print("Attempting to create mesh from point cloud...")
        # Estimate normals
        combined_pcd.estimate_normals()
        
        # Create mesh using Poisson reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            combined_pcd, depth=9)
        
        # Save mesh
        mesh_path = os.path.join(output_dir, "scene_mesh.obj")
        o3d.io.write_triangle_mesh(mesh_path, mesh)
        print(f"Scene mesh saved to {mesh_path}")
    except Exception as e:
        print(f"Error creating mesh: {e}")
        print("Mesh creation skipped. Try with a denser point cloud or adjust parameters.")

if __name__ == "__main__":
    main()