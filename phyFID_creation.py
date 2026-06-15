from PhyFID.utils import*
from lib.diffusion_lib.Logger import Logger


DATA_ROOT = "data/RT28"  # ou data/MNIST
TRAIN_PT = os.path.join(DATA_ROOT, "processed", "o_training.pt")
TEST_PT = os.path.join(DATA_ROOT, "processed", "o_test.pt")
VAL_PT = os.path.join(DATA_ROOT, "processed", "o_validation.pt")


PHYFID_DEMO_ENCODER = "outputs/phyFID/phyfid_demo_encoder.pt"
PHYFID_DEMO_STATS = "outputs/phyFID//phyfid_demo_val_stats.npz"
PHYFID_DEMO_LOG = "outputs/phyFID//phyfid_demo_training.log"
PHYFID_DEMO_CSV = "outputs/phyFID//phyfid_demo_training.csv"

PHYFID_DEMO_EPOCHS = 200
PHYFID_DEMO_FEATURE_DIM = 32
PHYFID_DEMO_BATCH_SIZE = 128

MAX_DEMO_TRAIN_SIZE = 1024
MAX_EVAL_SIZE = 100


phyfid_logger = Logger(PHYFID_DEMO_LOG, PHYFID_DEMO_CSV, name="phyfid_demo_logger")


build_phyfid_reference(
    dataset_train_path=TRAIN_PT,
    dataset_val_path=VAL_PT,
    encoder_path=PHYFID_DEMO_ENCODER,
    stats_path=PHYFID_DEMO_STATS,
    train_epochs=PHYFID_DEMO_EPOCHS,
    feature_dim=PHYFID_DEMO_FEATURE_DIM,
    batch_size=PHYFID_DEMO_BATCH_SIZE,
    max_train_size=MAX_DEMO_TRAIN_SIZE,
    max_val_size=MAX_EVAL_SIZE,
    logger=phyfid_logger,
)
