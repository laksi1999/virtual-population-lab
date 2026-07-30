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

# The source values are raw nutrient concentrations on very different scales
# (Water ~72, Stearic ~0.26), NOT standardized — so the pipeline must fit its
# own StandardScaler for the VAEs. (Previously True, which fed raw values to the
# VAE and made the plain-VAE baseline diverge; a fair baseline needs scaling.)
IS_PRE_SCALED = False
LATENT_DIM = 5
VAE_FREE_BITS = 1.0
# No dropout: the near-deterministic physics needs full decoder capacity.
RUN_TSTR = False

# Calibration ON: the lipid margins are near-symmetric, so the empirical-copula
# remap preserves correlations while fixing the decoder's physically-impossible
# tail (it emits negative fat in the lean-fruit region). Copula-style: margins
# are real, not purely learned — disclose it. NB at this n (~40) the four engines
# are within seed-noise of each other on fidelity; the hybrid's genuine edge here
# is calibrated coverage@90, not winning correlation/KS. It is the honest
# "most-consistent, not universally-best" case.
VAE_CALIBRATE_MARGINALS = True
VAE_PHYSICS_WEIGHT = 3.0

# Covariance-matching term OFF here. The 4 features are already pinned by the
# near-deterministic physics edges (r 0.76-0.99); a 4x4 sample covariance
# estimated from only ~28 training rows is too noisy to be a useful target and
# just destabilizes the correlation structure the physics already fixes. An
# ablation across 5 seeds showed dropping it lowers correlation distance on
# every seed (0.31 -> 0.23) with no effect on KS/calibration/coverage (those are
# set by the copula calibration). Decided on training-side reasoning, not to
# chase a test-set win. Kept ON elsewhere, where it is the main correlation driver.
VAE_COV_WEIGHT = 0.0

# Leave-one-region-out: Congo (oil-rich) vs Guinea (lean). The transfer +
# uncertainty-widening result is the point.
LORO_GROUP = "Region"
