import logging
from datetime import datetime
import os

def setup_logger():
    os.makedirs("Debug_logs", exist_ok=True)
    log_file = f'Debug_logs/app_debug_{datetime.now().strftime("%Y%m%d_%H%M")}.log'
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger