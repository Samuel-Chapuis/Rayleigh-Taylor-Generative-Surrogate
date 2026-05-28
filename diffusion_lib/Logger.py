import logging
import os
from datetime import datetime

def setup_logger(log_dir: str = "outputs/logs"):
    """
    Verifie que le dossier log_dir existe ( et le creer sinon ) puis genere un fichier .csv qui permet de sauvegardé l'entrainement du modele.

    Args:
        log_dir (str, optional): The directory where log files will be stored. Defaults to "outputs/logs".
    """    

    os.makedirs(log_dir, exist_ok=True)
    log_filename = f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    log_path = os.path.join(log_dir, log_filename)

    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    logging.info("epoch,loss")


def log(epoch: int, loss: float):
    """
    Enregistre une entrée de log avec l'époque et la perte correspondante.

    Args:
        epoch (int): The current epoch number.
        loss (float): The loss value to log.
    """    
    logging.info(f"{epoch},{loss}")