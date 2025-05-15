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
from networks.depth_decoder import DepthDecoder
from networks.resnet_encoder import ResnetEncoder
from utils import output_to_depth
from visualize_pc import generate_pointcloud

def parse_args():
    parser = argparse.ArgumentParser(description="Image Folder to Point Cloud converter")
    
    parser.add_argument("--folder", type=str, required=True,
                        help="Path to the input image folder")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: creates 'pc_output' in the current directory)")
    parser.add_argument("--ckpt_path", type=str, default="./ckpts",
                        help="Path to the checkpoints directory containing encoder.pth and depth.pth")
    parser.add_argument("--focal_length", type=float, default=886.81,
                        help="Focal length of the camera")
    parser.add_argument("--center_x", type=float, default=512.0, 
                        help="Principal point x-coordinate (will be overridden by actual image dimensions unless forced)")
    parser.add_argument("--center_y", type=float, default=384.0,
                        help="Principal point y-coordinate (will be overridden by actual image dimensions unless forced)")
    parser.add_argument("--width", type=int, default=1024,
                        help="Camera width (will be overridden by actual image dimensions unless forced)")
    parser.add_argument("--height", type=int, default=768, 
                        help="Camera height (will be overridden by actual image dimensions unless forced)")
    parser.add_argument("--force_camera_params", action="store_true",
                        help="Force using the specified camera parameters instead of image dimensions")
    parser.add_argument("--min_depth", type=float, default=0.1,
                        help="Minimum depth value")
    parser.add_argument("--max_depth", type=float, default=10.0,
                        help="Maximum depth value")
    parser.add_argument("--save_depth", action="store_true",
                        help="Save depth map as PNG and NPY files")
    parser.add_argument("--input_size", type=int, default=256,
                        help="Size for neural network input (default: 256)")
    parser.add_argument("--output_size", type=int, default=512,
                        help="Size for neural network output (default: 512)")
    parser.add_argument("--no_cuda", action="store_true",
                        help="If set, disables CUDA even if available")
    parser.add_argument("--image_ext", type=str, default="jpg,jpeg,png",
                        help="Comma-separated list of image extensions to process (default: jpg,jpeg,png)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip processing images that already have corresponding point clouds")
    parser.add_argument("--batch_size", type=int, default=0,
                        help="Number of images to process in parallel (default: auto-determined based on device)")
    parser.add_argument("--no_batch", action="store_true",
                        help="Disable batch processing and process images one by one")
    
    return parser.parse_args()

def setup_camera_params(args, width, height):
    """Set global camera parameters for point cloud generation"""
    import visualize_pc
    visualize_pc.focalLength = args.focal_length
    visualize_pc.centerX = args.center_x if args.force_camera_params else width / 2
    visualize_pc.centerY = args.center_y if args.force_camera_params else height / 2
    visualize_pc.intWidth = args.width if args.force_camera_params else width
    visualize_pc.intHeight = args.height if args.force_camera_params else height

def process_image(image_path, device, encoder, depth_decoder, input_size, output_size, min_depth, max_depth):
    """Process a single image and return the depth map"""
    # Read image
    img = cv2.imread(image_path, -1)
    if img is None:
        return None, None, None
        
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
    
    return metric_depth, original_width, original_height


def process_image_batch(image_paths, device, encoder, depth_decoder, input_size, output_size, min_depth, max_depth, batch_size=4):
    """Process a batch of images in parallel and return their depth maps"""
    results = []
    
    # Process in batches
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        original_sizes = []
        valid_indices = []
        
        # Read and preprocess each image in the batch
        for idx, path in enumerate(batch_paths):
            img = cv2.imread(path, -1)
            if img is None:
                results.append((path, None, None, None))
                continue
                
            # Track valid images and their original sizes
            valid_indices.append(idx)
            original_sizes.append((img.shape[0], img.shape[1]))  # height, width
            
            # Preprocess
            raw_img = np.transpose(img[:, :, :3], (2, 0, 1))
            input_tensor = torch.from_numpy(raw_img).float() / 255.0
            batch_images.append(input_tensor)
        
        if not batch_images:
            continue
            
        # Stack into a batch tensor
        input_batch = torch.stack(batch_images).to(device)
        
        # Resize for network input
        input_batch = torch.nn.functional.interpolate(
            input_batch, (input_size, input_size), mode="bilinear", align_corners=False
        )
        
        # Forward pass (single network inference for multiple images)
        with torch.no_grad():
            features = encoder(input_batch)
            outputs = depth_decoder(features)
        
        # Process each output
        for idx, (original_height, original_width) in enumerate(original_sizes):
            # Get depth prediction for this image
            out = outputs[("out", 0)][idx:idx+1]
            
            # Resize to original dimensions
            out_resized = torch.nn.functional.interpolate(
                out, (original_height, original_width), mode="bilinear", align_corners=False
            )
            
            # Convert to depth
            depth = output_to_depth(out_resized, min_depth, max_depth)
            metric_depth = depth.detach().cpu().numpy().squeeze()
            
            # Store result with original path
            path_idx = valid_indices[idx]
            results.append((batch_paths[path_idx], metric_depth, original_width, original_height))
    
    # Sort results by original order of paths
    path_to_idx = {path: idx for idx, path in enumerate(image_paths)}
    results.sort(key=lambda x: path_to_idx.get(x[0], float('inf')))
    
    return results

def save_depth_visualization(metric_depth, output_path, min_depth, max_depth):
    """Save a visualization of the depth map"""
    normalizer = mpl.colors.Normalize(vmin=min_depth, vmax=max_depth)
    mapper = cm.ScalarMappable(norm=normalizer, cmap="turbo")
    colormapped_im = (mapper.to_rgba(metric_depth)[:, :, :3] * 255).astype(np.uint8)
    
    cv2.imwrite(output_path, colormapped_im[:,:,[2,1,0]])

def get_image_files(folder_path, extensions):
    """Get list of image files with specified extensions"""
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

def process_folder(args, encoder, depth_decoder, device):
    """Process all images in the folder using batch processing"""
    image_files = get_image_files(args.folder, args.image_ext)
    
    if not image_files:
        print(f"No images with extensions {args.image_ext} found in {args.folder}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Filter out already processed images if skip_existing is enabled
    if args.skip_existing:
        to_process = []
        skipped_files = []
        for image_file in image_files:
            base_filename = os.path.splitext(os.path.basename(image_file))[0]
            pc_path = os.path.join(args.output_dir, f"{base_filename}_pc.ply")
            if os.path.exists(pc_path):
                skipped_files.append(image_file)
            else:
                to_process.append(image_file)
        
        skipped_count = len(skipped_files)
        if skipped_count > 0:
            print(f"Skipping {skipped_count} images that already have point clouds")
        image_files = to_process
    else:
        skipped_count = 0
    
    if not image_files:
        print("No new images to process")
        return
    
    # Initialize counters
    processed_count = 0
    failed_count = 0
    
    # If no batch processing requested, use the original method
    if args.no_batch:
        print("Processing images one by one...")
        processed_count = 0
        for image_file in tqdm(image_files):
            base_filename = os.path.splitext(os.path.basename(image_file))[0]
            depth_npy_path = os.path.join(args.output_dir, f"{base_filename}_depth.npy")
            depth_png_path = os.path.join(args.output_dir, f"{base_filename}_depth.png")
            pc_path = os.path.join(args.output_dir, f"{base_filename}_pc.ply")
            
            # Process image to get depth
            metric_depth, width, height = process_image(
                image_file, device, encoder, depth_decoder, 
                args.input_size, args.output_size, args.min_depth, args.max_depth
            )
            
            if metric_depth is None:
                print(f"Error: Could not process image {image_file}")
                failed_count += 1
                continue
            
            # Update camera parameters for this specific image
            setup_camera_params(args, width, height)
            
            # Save depth as numpy array
            np.save(depth_npy_path, metric_depth)
            
            # Save depth visualization if requested
            if args.save_depth:
                save_depth_visualization(metric_depth, depth_png_path, args.min_depth, args.max_depth)
            
            # Generate point cloud
            generate_pointcloud(image_file, depth_npy_path, pc_path)
            processed_count += 1
        
        # Print summary
        print("\nProcessing complete!")
        print(f"Total images found: {len(image_files) + skipped_count}")
        print(f"Successfully processed: {processed_count}")
        print(f"Skipped (already exist): {skipped_count}")
        print(f"Failed to process: {failed_count}")
        return
        
    # Determine batch size based on user input or available memory
    if args.batch_size > 0:
        # Use user-specified batch size
        batch_size = min(args.batch_size, len(image_files))
    else:
        # Auto-determine based on device
        if torch.cuda.is_available() and not args.no_cuda:
            # Use larger batch size for GPU processing
            batch_size = min(4, len(image_files))  # Can be adjusted based on available GPU memory
        else:
            # Smaller batch size for CPU processing
            batch_size = min(2, len(image_files))
    
    print(f"Processing in batches of {batch_size} images")
    
    # Process images in batches
    from tqdm import tqdm
    
    # Process images in batches
    results = process_image_batch(
        image_files, device, encoder, depth_decoder,
        args.input_size, args.output_size, args.min_depth, args.max_depth,
        batch_size=batch_size
    )
    
    # Process results
    print("Generating point clouds...")
    for image_file, metric_depth, width, height in tqdm(results):
        base_filename = os.path.splitext(os.path.basename(image_file))[0]
        depth_npy_path = os.path.join(args.output_dir, f"{base_filename}_depth.npy")
        depth_png_path = os.path.join(args.output_dir, f"{base_filename}_depth.png")
        pc_path = os.path.join(args.output_dir, f"{base_filename}_pc.ply")
        
        if metric_depth is None:
            print(f"Error: Could not process image {image_file}")
            failed_count += 1
            continue
        
        # Update camera parameters for this specific image
        setup_camera_params(args, width, height)
        
        # Save depth as numpy array
        np.save(depth_npy_path, metric_depth)
        
        # Save depth visualization if requested
        if args.save_depth:
            save_depth_visualization(metric_depth, depth_png_path, args.min_depth, args.max_depth)
        
        # Generate point cloud
        generate_pointcloud(image_file, depth_npy_path, pc_path)
        processed_count += 1
    
    # Print summary
    print("\nProcessing complete!")
    print(f"Total images found: {len(image_files) + skipped_count}")
    print(f"Successfully processed: {processed_count}")
    print(f"Skipped (already exist): {skipped_count}")
    print(f"Failed to process: {failed_count}")

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
    args.output_dir = args.output if args.output else "pc_output"
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    print("Loading neural network...")
    # Load networks
    with torch.no_grad():
        # Load encoder
        encoder = ResnetEncoder(152, False)
        encoder_path = os.path.join(args.ckpt_path, "encoder.pth")
        loaded_dict_enc = torch.load(encoder_path, map_location=device)
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
        encoder.load_state_dict(filtered_dict_enc)
        encoder.to(device)
        encoder.eval()
        
        # Load depth decoder
        depth_decoder = DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))
        depth_path = os.path.join(args.ckpt_path, "depth.pth")
        loaded_dict = torch.load(depth_path, map_location=device)
        depth_decoder.load_state_dict(loaded_dict)
        depth_decoder.to(device)
        depth_decoder.eval()
        
        # Process all images in the folder
        process_folder(args, encoder, depth_decoder, device)

if __name__ == "__main__":
    main()