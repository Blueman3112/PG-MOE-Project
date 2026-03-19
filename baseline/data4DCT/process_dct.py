import os
import glob
import argparse
import logging
from datetime import datetime
import multiprocessing
from functools import partial

import numpy as np
import scipy.fftpack
from PIL import Image
from tqdm import tqdm

def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_file = os.path.join(output_dir, f"dct_conversion_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def dct2(a):
    """Compute 2D Discrete Cosine Transform"""
    return scipy.fftpack.dct(scipy.fftpack.dct(a, axis=0, norm='ortho'), axis=1, norm='ortho')

def process_single_image(img_path, output_dir):
    """Process a single image and save as .npy"""
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    save_path = os.path.join(output_dir, f"{base_name}.npy")
    
    # Skip if already processed
    if os.path.exists(save_path):
        return True
        
    try:
        # 1. Read, convert to grayscale, resize
        img = Image.open(img_path).convert('L').resize((256, 256))
        img_np = np.array(img)
        
        # 2. Compute 2D DCT
        dct_img = dct2(img_np)
        
        # 3. Log scale to highlight frequency features (add small constant to avoid log(0))
        dct_img = np.log(np.abs(dct_img) + 1e-12)
        
        # 4. Normalize to [0, 1] range
        dct_img = (dct_img - np.min(dct_img)) / (np.max(dct_img) - np.min(dct_img) + 1e-8)
        
        # 5. Save as float32 numpy array
        np.save(save_path, dct_img.astype(np.float32))
        return True
    except Exception as e:
        return f"Error processing {img_path}: {str(e)}"

def get_imglist(path):
    ext = [".jpg", ".bmp", ".png", ".jpeg", ".webp"]
    files = []
    for e in ext:
        files.extend(glob.glob(os.path.join(path, f'*{e}')))
        files.extend(glob.glob(os.path.join(path, f'*{e.upper()}')))
    return sorted(files)

def process_directory(input_dir, output_dir, num_workers, logger):
    os.makedirs(output_dir, exist_ok=True)
    all_imgnames_list = get_imglist(input_dir)
    
    # Filter out already processed images
    imgnames_list = []
    for imgpath in all_imgnames_list:
        base_name = os.path.splitext(os.path.basename(imgpath))[0]
        save_path = os.path.join(output_dir, f"{base_name}.npy")
        if not os.path.exists(save_path):
            imgnames_list.append(imgpath)
            
    num_items = len(imgnames_list)
    total_items = len(all_imgnames_list)
    
    logger.info(f"Processing directory: {input_dir} -> {output_dir}")
    logger.info(f"Found {total_items} total images. {total_items - num_items} already processed. {num_items} remaining.")
    
    if num_items == 0:
        return

    # Create a partial function with the fixed output_dir argument
    process_func = partial(process_single_image, output_dir=output_dir)
    
    # Use multiprocessing Pool
    success_count = 0
    error_count = 0
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        # Use tqdm to show progress bar, updating it as tasks complete
        results = list(tqdm(pool.imap(process_func, imgnames_list), total=num_items, desc=os.path.basename(input_dir)))
        
        for res in results:
            if res is True:
                success_count += 1
            else:
                error_count += 1
                logger.error(res)
                
    logger.info(f"Completed {os.path.basename(input_dir)}: {success_count} successful, {error_count} failed.")

def main():
    parser = argparse.ArgumentParser(description="Convert images to DCT frequency representations")
    parser.add_argument("--dataset_root", type=str, required=True, help="Root directory of the dataset (e.g., dataset-A)")
    parser.add_argument("--output_root", type=str, required=True, help="Root directory for output (e.g., dataset-A-DCT)")
    parser.add_argument("--num_workers", type=int, default=multiprocessing.cpu_count(), help="Number of CPU workers for multiprocessing")
    args = parser.parse_args()

    # Setup logger
    logger = setup_logging(args.output_root)
    logger.info("Starting DCT conversion process...")
    logger.info(f"Dataset Root: {args.dataset_root}")
    logger.info(f"Output Root: {args.output_root}")
    logger.info(f"Workers: {args.num_workers}")

    # Define subdirectories structure
    subdirs = [
        'train/real', 'train/fake',
        'val/real', 'val/fake',
        'test/real', 'test/fake'
    ]

    for subdir in subdirs:
        input_dir = os.path.join(args.dataset_root, subdir)
        output_dir = os.path.join(args.output_root, subdir)
        
        if os.path.exists(input_dir):
            process_directory(input_dir, output_dir, args.num_workers, logger)
        else:
            logger.warning(f"Input directory not found: {input_dir}")

    logger.info("All DCT processing completed.")

if __name__ == "__main__":
    main()