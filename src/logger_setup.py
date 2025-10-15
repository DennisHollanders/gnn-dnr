import logging
import os
from datetime import datetime
from pathlib import Path
import multiprocessing
from logging.handlers import QueueHandler, QueueListener


def setup_logging(
    log_dir: str | Path = "data_generation/logs",
    run_tag: str | None = None,
    level_file: int = logging.DEBUG,
    level_console: int = logging.DEBUG,
) -> tuple[QueueHandler, QueueListener]:  

    if run_tag is None:
        run_tag = os.environ.get("RUN_TAG")
    if run_tag is None:
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.environ["RUN_TAG"] = run_tag

    log_dir = Path(log_dir).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)
    logfile = log_dir / f"{run_tag}.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()  

    fmt_file = logging.Formatter(
        "%(asctime)s  %(processName)s  %(levelname)-8s  "
        "%(name)s:%(lineno)d  %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    fmt_cons = logging.Formatter(
        "%(asctime)s  %(levelname)-7s  %(message)s", "%H:%M:%S"
    )

    log_queue = multiprocessing.Queue(-1) 
    queue_handler = QueueHandler(log_queue) 
    queue_handler.setLevel(logging.DEBUG)
    root.addHandler(queue_handler)  

    file_handler = logging.FileHandler(logfile, mode="a", encoding="utf-8")
    file_handler.setLevel(level_file)
    file_handler.setFormatter(fmt_file)

    console_handler = logging.StreamHandler() 
    console_handler.setLevel(level_console)
    console_handler.setFormatter(fmt_cons)

    # Listener for queue
    queue_listener = QueueListener(log_queue, file_handler, console_handler)  
    queue_listener.daemon = True
    queue_listener.start()
    root._onefile_configured = True
    return queue_handler, queue_listener