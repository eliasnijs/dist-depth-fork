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
from networks.depth_decoder import DepthDecoder
from networks.resnet_encoder import ResnetEncoder
from utils import output_to_depth
from visualize_pc import generate_pointcloud

def parse_args():
    parser = argparse.ArgumentParser(description="Image to Point Cloud converter")

    parser.add_argument("--image", type=str, required=True,
                        help="Path to the input image")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: creates 'pc_output' in the current directory)")
    parser.add_argument("--ckpt_path", type=str, default="./ckpts",
                        help="Path to the checkpoints directory containing encoder.pth and depth.pth")
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

    return parser.parse_args()

def setup_camera_params(args):
    """Set global camera parameters for point cloud generation"""
    import visualize_pc
    visualize_pc.focalLength = args.focal_length
    visualize_pc.centerX = args.center_x
    visualize_pc.centerY = args.center_y
    visualize_pc.intWidth = args.width
    visualize_pc.intHeight = args.height

def process_image(image_path, device, encoder, depth_decoder, input_size, output_size, min_depth, max_depth):
    """Process an image and return the depth map"""
    # Read image
    img = cv2.imread(image_path, -1)
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
    metric_depth = depth.cpu().numpy().squeeze()

    return metric_depth

def save_depth_visualization(metric_depth, output_path, min_depth, max_depth):
    """Save a visualization of the depth map"""
    normalizer = mpl.colors.Normalize(vmin=min_depth, vmax=max_depth)
    mapper = cm.ScalarMappable(norm=normalizer, cmap="turbo")
    colormapped_im = (mapper.to_rgba(metric_depth)[:, :, :3] * 255).astype(np.uint8)

    cv2.imwrite(output_path, colormapped_im[:,:,[2,1,0]])

def main():
    args = parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")

    # Create output directory if needed
    output_dir = args.output if args.output else "pc_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read image dimensions to update camera parameters
    img = cv2.imread(args.image, -1)
    if img is None:
        print(f"Error: Could not read image {args.image}")
        return

    height, width = img.shape[:2]

    # Override width and height with actual image dimensions if not explicitly set
    if args.width == 1024 and args.height == 768:  # If using defaults
        args.width = width
        args.height = height
        # Adjust center points to match image dimensions
        args.center_x = width / 2
        args.center_y = height / 2
        print(f"Using image dimensions: {width}x{height} for camera parameters")

    # Setup camera parameters for point cloud generation
    setup_camera_params(args)

    # Get base filename without extension
    base_filename = os.path.splitext(os.path.basename(args.image))[0]
    depth_npy_path = os.path.join(output_dir, f"{base_filename}_depth.npy")
    depth_png_path = os.path.join(output_dir, f"{base_filename}_depth.png")
    pc_path = os.path.join(output_dir, f"{base_filename}_pc.ply")

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

        print(f"Processing {args.image}...")
        # Process image to get depth
        metric_depth = process_image(
            args.image, device, encoder, depth_decoder,
            args.input_size, args.output_size, args.min_depth, args.max_depth
        )

        # Save depth as numpy array
        np.save(depth_npy_path, metric_depth)
        print(f"Depth map saved to {depth_npy_path}")

        # Save depth visualization if requested
        if args.save_depth:
            save_depth_visualization(metric_depth, depth_png_path, args.min_depth, args.max_depth)
            print(f"Depth visualization saved to {depth_png_path}")

        # Generate point cloud
        print(f"Generating point cloud...")
        generate_pointcloud(args.image, depth_npy_path, pc_path)
        print(f"Point cloud saved to {pc_path}")

if __name__ == "__main__":
    main()
