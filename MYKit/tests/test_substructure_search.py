from graph_mol import Mol
import numpy as np

benzene = Mol(
    ["C", "C", "C", "C", "C", "C"],
    [
        (0, 1, 3),
        (1, 2, 3),
        (2, 3, 3),
        (3, 4, 3),
        (4, 5, 3),
        (5, 0, 3),
    ],
)

benzene_fp = benzene.fingerprint()
onbits = np.nonzero(benzene_fp)[0]

benzene_with_cabbage = Mol(
    ["C", "C", "C", "C", "C", "C", "F", "C"],
    [
        (0, 1, 3),
        (1, 2, 3),
        (2, 3, 3),
        (3, 4, 3),
        (4, 5, 3),
        (5, 0, 3),
        (6, 0, 1),
        (7, 1, 1),
    ],
)

benzene_with_cabbage_fp = benzene_with_cabbage.fingerprint()

for bit in onbits:
    assert benzene_with_cabbage_fp[bit] == 1

cyclohexane = Mol(
    [
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
    ],
    [
        (0, 1, 1),
        (1, 2, 1),
        (2, 3, 1),
        (3, 4, 1),
        (4, 5, 1),
        (5, 0, 1),
    ],
)

cyclohexane_fp = cyclohexane.fingerprint()

assert not all(cyclohexane_fp[bit] == 1 for bit in onbits)
