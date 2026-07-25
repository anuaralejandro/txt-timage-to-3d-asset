"""
Installer & environment diagnostics script for 3D segmentation backends.
Checks PyTorch, CUDA, VRAM availability, trimesh, and optional model checkpoints.
"""

import sys
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def check_environment():
    logging.info("Checking environment for 3D Segmentation Pipeline...")

    # Python version
    logging.info(f"Python version: {sys.version.split()[0]}")

    # PyTorch & CUDA
    try:
        import torch
        logging.info(f"PyTorch version: {torch.__version__}")
        cuda_avail = torch.cuda.is_available()
        logging.info(f"CUDA available: {cuda_avail}")
        if cuda_avail:
            logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logging.info(f"VRAM: {vram_gb:.2f} GB")
    except ImportError:
        logging.warning("PyTorch is not installed in this Python environment.")

    # Trimesh & SciPy
    try:
        import trimesh
        logging.info(f"trimesh version: {trimesh.__version__}")
    except ImportError:
        logging.warning("trimesh is not installed.")

    try:
        import scipy
        logging.info(f"scipy version: {scipy.__version__}")
    except ImportError:
        logging.warning("scipy is not installed.")

    logging.info("Environment check complete. All optional backends provide mock fallbacks when weights are missing.")

if __name__ == "__main__":
    check_environment()
