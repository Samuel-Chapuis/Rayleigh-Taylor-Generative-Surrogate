from lib.PhyFID.utils import*
from lib.diffusion_lib.Logger import Logger


FEATURE = 64 # Taille de l'espace des features souvent notee d, et on prend comme taille de data pour l'entrainement d*100 ou d*200.

DATA_ROOT = "data/RT28"  # ou data/MNIST
TRAIN_PT = os.path.join(DATA_ROOT, "processed", "training.pt")
TEST_PT = os.path.join(DATA_ROOT, "processed", "test.pt")
VAL_PT = os.path.join(DATA_ROOT, "processed", "validation.pt")


PHYFID_ENCODER = "outputs/phyFID/phyfid_encoder.pt"
PHYFID_STATS = "outputs/phyFID//phyfid_val_stats.npz"
PHYFID_LOG = "outputs/phyFID//phyfid_training.log"
PHYFID_CSV = "outputs/phyFID//phyfid_training.csv"

PHYFID_EPOCHS = 600
PHYFID_BATCH_SIZE = 128

MAX_EVAL_SIZE = 100


phyfid_logger = Logger(PHYFID_LOG, PHYFID_CSV, name="phyfid_logger")


build_phyfid_reference(
    dataset_train_path=TRAIN_PT,
    dataset_val_path=VAL_PT,
    encoder_path=PHYFID_ENCODER,
    stats_path=PHYFID_STATS,
    train_epochs=PHYFID_EPOCHS,
    feature_dim=FEATURE,
    batch_size=PHYFID_BATCH_SIZE,
    max_train_size=FEATURE*200,
    max_val_size=MAX_EVAL_SIZE,
    logger=phyfid_logger,
)
