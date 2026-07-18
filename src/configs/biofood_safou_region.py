"""
Safou (Dacryodes edulis) lipid composition — the project's only individual-level
nutrient population with real growing regions. 41 individual African-pear fruits
(one fruit per tree): Congo (oil-rich, n=21) and Guinea (lean, n=20). Near-
deterministic mass-balance graph (Water->Fat r=-0.99; Fat->fatty acids). Small n,
so treat fidelity numbers as indicative. 4 features = 3 nutrients (fat, palmitic,
stearic) + water.

Run: make run CONFIG=biofood_safou_region ; make loro CONFIG=biofood_safou_region
"""
from src.configs.base import *  # noqa: F401,F403

DATA_PATH = "data/cleaned-biofood-safou-region.csv"
DATA_SOURCE_URL = "https://www.fao.org/infoods/infoods/tables-and-databases/faoinfoods-databases/en/"

ID_COLUMNS = []
LABEL_COLUMN = "Region"

FEATURES = ["Water", "Fat", "Palmitic", "Stearic"]

# Universal mass-balance chain: water displaces fat; the fatty acids are
# fractions of total fat. Holds in both regions.
ROOT_VARIABLES = ["Water"]
CAUSAL_GRAPH = [
    ("Water", "Fat"),
    ("Fat", "Palmitic"),
    ("Fat", "Stearic"),
]

IS_PRE_SCALED = True
LATENT_DIM = 5
VAE_FREE_BITS = 1.0
# No dropout: the near-deterministic physics needs full decoder capacity.
RUN_TSTR = False

# Calibration ON: the lipid margins are near-symmetric, so the empirical-copula
# remap preserves correlations while fixing the decoder's physically-impossible
# tail (it emits negative fat in the lean-fruit region). With physics_weight=5
# on the near-deterministic edges, the calibrated hybrid leads on joint
# structure. Copula-style: margins are real, not purely learned — disclose it.
VAE_CALIBRATE_MARGINALS = True
VAE_PHYSICS_WEIGHT = 3.0

# Leave-one-region-out: Congo (oil-rich) vs Guinea (lean). The transfer +
# uncertainty-widening result is the point.
LORO_GROUP = "Region"
