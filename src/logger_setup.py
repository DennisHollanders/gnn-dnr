import logging
from queue import Queue
from logging.handlers import QueueHandler, QueueListener

log_queue = Queue(-1)
queue_handler = QueueHandler(log_queue)
file_handler = logging.FileHandler("run.log")
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
file_handler.setFormatter(formatter)
listener = QueueListener(log_queue, file_handler)
listener.start()

logger = logging.getLogger("Distribution Network Reconfiguration -- Dataset Generation")
logger.addHandler(queue_handler)
logger.setLevel(logging.INFO)
