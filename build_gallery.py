import os
import time
import glob
import logging
import asyncio
import cv2
import torch
import numpy as np

# Setup Godhaar imports
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from godhaar.model import GodhaarModel
from pipeline.yolo_crop import load_yolo, warmup_yolo, crop_cattle
from pipeline.quality import quality_check_cv2
from pipeline.muzzle import embed_batch
from faiss_index import FaissIndex

BATCH_SIZE = 64
MAX_IMAGES = 100000
IMAGE_DIR = r"D:\Group Projects\Godhaar\data\images"
MODEL_PATH = r"d:\Group Projects\Godhaar\Wildlife\for_aditya\best_top1.pt"
INDEX_PATH = r"d:\Group Projects\Godhaar\indexes\gallery_100k.index"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("gallery_builder")

def build_gallery():
    # 1. Initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")
    if device.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load Model
    log.info(f"Loading GodhaarModel from {MODEL_PATH}...")
    model, _ = GodhaarModel.load_checkpoint(MODEL_PATH, device=device)
    model.eval()
    
    # Load YOLO
    log.info("Loading YOLO...")
    load_yolo()
    warmup_yolo()
    
    # Init FAISS
    faiss_index = FaissIndex(embedding_dim=256)
    
    # 2. Collect 100k images
    log.info(f"Scanning {IMAGE_DIR} for images...")
    # Getting 100k image paths iteratively to save memory/time
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    
    # Using os.scandir/os.walk for faster traversal of huge directories
    image_files = []
    for root, _, files in os.walk(IMAGE_DIR):
        for f in files:
            if os.path.splitext(f)[1].lower() in valid_exts:
                image_files.append(os.path.join(root, f))
                if len(image_files) >= MAX_IMAGES:
                    break
        if len(image_files) >= MAX_IMAGES:
            break
            
    log.info(f"Found {len(image_files)} valid images to process. Starting pipeline...")

    # Metrics
    failed_quality = 0
    failed_yolo = 0
    total_embed_time = 0.0
    total_faiss_time = 0.0
    processed_images = 0
    
    batch_crops = []
    
    start_total = time.time()
    
    for i, img_path in enumerate(image_files):
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            failed_quality += 1
            continue
            
        crop, det_status, _ = crop_cattle(img_bgr)
        if crop is None:
            failed_yolo += 1
            continue
            
        q_status, _ = quality_check_cv2(crop)
        if q_status != "GOOD":
            failed_quality += 1
            continue
            
        # pipeline/muzzle.py embed_batch takes bytes.
        # Let's convert crop to bytes as embed_batch currently expects bytes.
        # (It uses cv2.imencode inside the server pipeline).
        crop_bytes = cv2.imencode(".jpg", crop)[1].tobytes()
        batch_crops.append(crop_bytes)
        processed_images += 1
        
        # Process batch
        if len(batch_crops) >= BATCH_SIZE or i == len(image_files) - 1:
            if not batch_crops:
                continue
                
            # Embedding (GPU bound)
            t0 = time.time()
            embeddings = embed_batch(batch_crops, model, device)
            embed_duration = time.time() - t0
            total_embed_time += embed_duration
            
            # FAISS (CPU bound)
            t1 = time.time()
            emb_np = embeddings.numpy()
            asyncio.run(faiss_index.add_batch(emb_np))
            total_faiss_time += (time.time() - t1)
            
            batch_crops = []
            
        if (i + 1) % 1000 == 0:
            log.info(f"Processed {i+1}/{len(image_files)} images... (Faiss vectors: {len(faiss_index)})")
            
    # 3. Save index
    log.info(f"Saving FAISS index to {INDEX_PATH}...")
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    asyncio.run(faiss_index.save(INDEX_PATH))
    
    end_total = time.time()
    
    # 4. Report
    log.info("========== GALLERY BUILD REPORT ==========")
    log.info(f"Total Images Scanned:       {len(image_files)}")
    log.info(f"Successfully Embedded:      {processed_images}")
    log.info(f"Failed YOLO Detections:     {failed_yolo}")
    log.info(f"Failed Quality Checks:      {failed_quality}")
    log.info(f"Total Embedding Time:       {total_embed_time:.2f}s")
    if processed_images > 0:
        log.info(f"Average Embed Time/Image:   {(total_embed_time/processed_images)*1000:.2f}ms")
    log.info(f"Total FAISS Add Time:       {total_faiss_time:.2f}s")
    log.info(f"Total Script Run Time:      {end_total - start_total:.2f}s")
    log.info("==========================================")

if __name__ == "__main__":
    build_gallery()
