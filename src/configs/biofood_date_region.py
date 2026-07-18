"""
WITHIN-ONE-FRUIT leave-one-REGION-out: date (Phoenix dactylifera) by origin.

The Pathway-2 equity experiment: a single fruit (date), subgrouped by growing
region, to test transfer from a data-rich region to a data-scarce one and show
whether the model's uncertainty widens when it extrapolates off its training
support.

  77 date entries (complete-case on the 5 dense features Water + K + Fe + Ca +
  Mg), across three origins:
    - UAE     (n=64) — data-RICH
    - Tunisia (n=10) — data-SCARCE  (Maghreb, far from the Gulf: standardized
                       centroid distance to UAE = 3.1 sigma, genuine extrapolation)
    - Oman    (n=3)  — too small to hold out; stays in training (MIN_GROUP_SIZE=8)

Honest scope: only two regions are big enough to hold out (UAE, Tunisia), so
this is a single in/out split, not a multi-distance gradient — but it is a real
within-single-fruit, real-geography leave-one-region-out with the equity framing
(rich Gulf -> scarce Maghreb). The dilution physics still holds within date
(Water->Mg r=-0.77, ->K -0.75), so the physics engine is exercised.

Run: make run  CONFIG=biofood_date_region   (comparison on the pooled date data)
     make loro CONFIG=biofood_date_region   (leave-one-region-out transfer)
"""
from src.configs.base import *  # noqa: F401,F403 — re-exported as this module's own config values

DATA_PATH = "data/cleaned-biofood-date-region.csv"
DATA_SOURCE_URL = "https://www.fao.org/infoods/infoods/tables-and-databases/faoinfoods-databases/en/"

ID_COLUMNS = []
LABEL_COLUMN = "Region"

FEATURES = ["Water", "Potassium", "Iron", "Calcium", "Magnesium"]

# Water dilutes the per-100g mineral concentration — holds within date too.
ROOT_VARIABLES = ["Water"]
CAUSAL_GRAPH = [
    ("Water", "Potassium"),
    ("Water", "Iron"),
    ("Water", "Calcium"),
    ("Water", "Magnesium"),
]

IS_PRE_SCALED = False
LATENT_DIM = 3
VAE_FREE_BITS = 1.0
# Small n (53 train): cap epochs (shorter KL warmup) and add mild dropout so the
# deep engines don't overfit — 5000 epochs / no dropout degraded the hybrid here.
VAE_EPOCHS = 2000
VAE_DROPOUT = 0.1
RUN_TSTR = False

# Leave-one-region-out: UAE (rich) vs Tunisia (scarce); Oman too small, kept in
# training. The transfer/uncertainty result is the point of this config.
LORO_GROUP = "Region"
RANDOM_SEED = 42