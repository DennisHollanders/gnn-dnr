# logging_config.py
import logging
import sys
from pathlib import Path

def setup_logging(log_level=logging.INFO, log_file=None, module_name="dnr_optimizer"):
    logger = logging.getLogger(module_name)
    logger.setLevel(log_level)
    logger.propagate = False
    if logger.handlers:
        logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
    logging.getLogger("pyomo").setLevel(logging.WARNING)
    
    return logger

def get_logger(name=None, parent="dnr_optimizer"):
    if name:
        return logging.getLogger(f"{parent}.{name}")
    return logging.getLogger(parent)