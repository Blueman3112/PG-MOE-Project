
import os
import sys
import glob
import argparse
import torch
import cv2
import numpy as np
import PIL.Image
from torchvision import transforms

# Add the current directory to sys.path so it can find 'models'
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from models import build_model
import logging
from datetime import datetime

# Setup logging
def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_file = os.path.join(output_dir, f"conversion_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

processimg = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def read_batchimg(imgpath_list):
    img_list = []
    for imgpath in imgpath_list:
        try:
            img = PIL.Image.open(imgpath).convert('RGB')
            img_list.append(torch.unsqueeze(processimg(img), 0))
        except Exception as e:
            logging.warning(f"Failed to read image {imgpath}: {e}")
    if not img_list:
        return None
    return torch.cat(img_list, 0)

def normlize_np(img):
    img -= img.min()
    if img.max() != 0:
        img /= img.max()
    return img * 255.

def get_imglist(path):
    ext = [".jpg", ".bmp", ".png", ".jpeg", ".webp"]
    files = []
    for e in ext:
        files.extend(glob.glob(os.path.join(path, f'*{e}')))
        files.extend(glob.glob(os.path.join(path, f'*{e.upper()}')))
    return sorted(files)

def process_directory(model, input_dir, output_dir, batch_size, logger):
    os.makedirs(output_dir, exist_ok=True)
    all_imgnames_list = get_imglist(input_dir)
    
    # Filter out already processed images
    imgnames_list = []
    for imgpath in all_imgnames_list:
        save_name = os.path.basename(imgpath).rsplit('.', 1)[0] + '.png'
        save_path = os.path.join(output_dir, save_name)
        if not os.path.exists(save_path):
            imgnames_list.append(imgpath)
            
    num_items = len(imgnames_list)
    total_items = len(all_imgnames_list)
    
    logger.info(f"Processing directory: {input_dir} -> {output_dir}")
    logger.info(f"Found {total_items} total images. {total_items - num_items} already processed. {num_items} remaining.")
    
    if num_items == 0:
        return

    numnow = 0
    for mb_begin in range(0, num_items, batch_size):
        imgname_list = imgnames_list[mb_begin: min(mb_begin + batch_size, num_items)]
        imgs_np = read_batchimg(imgname_list)
        
        if imgs_np is None:
            continue
            
        try:
            img_cuda = imgs_np.cuda().to(torch.float32)
            img_cuda.requires_grad = True
            
            # Forward pass
            pre = model(img_cuda)
            model.zero_grad()
            
            # Backward pass to get gradients
            grad = torch.autograd.grad(pre.sum(), img_cuda, create_graph=False, retain_graph=False, allow_unused=False)[0]
            
            # Save gradients
            current_batch_size = len(imgname_list)
            for idx in range(current_batch_size):
                numnow += 1
                img_grad = normlize_np(grad[idx].permute(1, 2, 0).cpu().detach().numpy())
                save_name = os.path.basename(imgname_list[idx]).rsplit('.', 1)[0] + '.png'
                save_path = os.path.join(output_dir, save_name)
                cv2.imwrite(save_path, img_grad[..., ::-1])
                
                if numnow % 100 == 0:
                    logger.info(f"Processed {numnow}/{num_items} images in {os.path.basename(input_dir)}")
                    
        except Exception as e:
            logger.error(f"Error processing batch starting at {mb_begin}: {e}")
            torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True, help="Root directory of the dataset (e.g., dataset-A)")
    parser.add_argument("--output_root", type=str, required=True, help="Root directory for output (e.g., dataset-A-LGrad)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained model (.pth)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")
    args = parser.parse_args()

    # Setup logger
    logger = setup_logging(args.output_root)
    logger.info("Starting gradient generation process...")
    logger.info(f"Dataset Root: {args.dataset_root}")
    logger.info(f"Output Root: {args.output_root}")
    logger.info(f"Model Path: {args.model_path}")
    logger.info(f"Batch Size: {args.batch_size}")

    # Load model
    try:
        model = build_model(gan_type='stylegan',
                          module='discriminator',
                          resolution=256,
                          label_size=0,
                          image_channels=3,
                          minibatch_std_group_size=1) # Fixed: Set group size to 1 to handle arbitrary batch sizes
        model.load_state_dict(torch.load(args.model_path), strict=True)
        model.cuda()
        model.eval()
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

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
            process_directory(model, input_dir, output_dir, args.batch_size, logger)
        else:
            logger.warning(f"Input directory not found: {input_dir}")

    logger.info("All processing completed.")

if __name__ == "__main__":
    main()
