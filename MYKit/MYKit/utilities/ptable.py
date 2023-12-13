import pandas as pd
from MYKit import MYKit_ROOT

ptable = pd.read_csv(MYKit_ROOT / "utilities" / "ptable.csv")

ELEMENT_DICT = ptable.set_index("Symbol").to_dict(orient="index")
ATOMIC_NUM_DICT = ptable.set_index("AtomicNumber").to_dict(orient="index")
