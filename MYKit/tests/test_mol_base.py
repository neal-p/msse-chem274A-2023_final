import pytest
import numpy as np
from pathlib import Path
import sys

mykit_path = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(mykit_path))
from MYKit import Mol


###################################################################
# Define some structures
###################################################################

benzene_elements = ["C"] * 6
benzene_bonds = [(0, 1, 2), (1, 2, 1), (2, 3, 2), (3, 4, 1), (4, 5, 2), (5, 0, 1)]

toluene_elements = ["C"] * 7
toluene_bonds = [
    (0, 1, 2),
    (1, 2, 1),
    (2, 3, 2),
    (3, 4, 1),
    (4, 5, 2),
    (5, 0, 1),
    (0, 6, 1),
]

propane_elements = ["C"] * 3
propane_bonds = [(0, 1, 1), (1, 2, 1)]

cyclo_propane_elements = ["C"] * 3
cyclo_propane_bonds = [(0, 1, 1), (1, 2, 1), (2, 0, 1)]


###################################################################
# Write fingerprints once to check consistency between runs
###################################################################

benzene = Mol(benzene_elements, benzene_bonds)
print(benzene)
# toluene = Mol(toluene_elements, toluene_bonds)
# propane = Mol(propane_elements, propane_bonds)
# cyclopropane = Mol(cyclo_propane_elements, cyclo_propane_bonds)

# np.save("benzene_fp.npy", benzene.fingerprint())
# np.save("toluene_fp.npy", toluene.fingerprint())
# np.save("propane_fp.npy", propane.fingerprint())
# np.save("cyclo_propane_fp.npy", cyclopropane.fingerprint())


###################################################################
# Tests
###################################################################


@pytest.mark.parametrize(
    "elements, bonds, expected",
    [
        (benzene_elements, benzene_bonds, [[0, 1, 2, 3, 4, 5]]),
        (toluene_elements, toluene_bonds, [[0, 1, 2, 3, 4, 5]]),
        (propane_elements, propane_bonds, []),
        (cyclo_propane_elements, cyclo_propane_bonds, [[0, 1, 2]]),
    ],
)
def test_rings(elements, bonds, expected):
    mol = Mol(elements, bonds)
    assert [
        sorted(list(ring)) for ring in mol.rings
    ] == expected, "Rings not identified"


@pytest.mark.parametrize(
    "elements, bonds, expected",
    [
        (benzene_elements, benzene_bonds, 1),
        (toluene_elements, toluene_bonds, 1),
        (propane_elements, propane_bonds, 0),
        (cyclo_propane_elements, cyclo_propane_bonds, 1),
    ],
)
def test_numRings(elements, bonds, expected):
    mol = Mol(elements, bonds)
    assert mol.numRings == expected, "Wrong number of rings identified"


@pytest.mark.parametrize(
    "elements, bonds, expected",
    [
        (benzene_elements, benzene_bonds, [6]),
        (toluene_elements, toluene_bonds, [6]),
        (propane_elements, propane_bonds, []),
        (cyclo_propane_elements, cyclo_propane_bonds, [3]),
    ],
)
def test_ringSizes(elements, bonds, expected):
    mol = Mol(elements, bonds)
    assert mol.ringSizes == expected, "Wrong number of rings identified"


@pytest.mark.parametrize(
    "elements, bonds, expected",
    [
        (benzene_elements, benzene_bonds, "with 6 atoms, 6 bonds, and 1 rings"),
        (toluene_elements, toluene_bonds, "with 7 atoms, 7 bonds, and 1 rings"),
        (propane_elements, propane_bonds, "with 3 atoms, 2 bonds, and 0 rings"),
        (
            cyclo_propane_elements,
            cyclo_propane_bonds,
            "with 3 atoms, 3 bonds, and 1 rings",
        ),
    ],
)
def test_print(elements, bonds, expected):
    mol = Mol(elements, bonds)
    assert expected in str(mol), "Name not expected"

    name = "testname"
    mol.attributes["_Name"] = name

    assert name in str(mol), "Name not expected!"


@pytest.mark.parametrize(
    "elements, bonds, expected",
    [
        (benzene_elements, benzene_bonds, [0, 1, 2, 4, 5]),
        (toluene_elements, toluene_bonds, [0, 1, 2, 4, 5, 6]),
        (propane_elements, propane_bonds, [0, 1, 2]),
        (cyclo_propane_elements, cyclo_propane_bonds, [0, 1, 2]),
    ],
)
def test_getAtomNeighborhood(elements, bonds, expected):
    mol = Mol(elements, bonds)
    neighbors = mol.getAtomNeighborhood(0, radius=2)
    assert sorted(neighbors) == expected, "Incorrect neighbors"


@pytest.mark.parametrize(
    "elements, bonds, length, expected",
    [
        (benzene_elements, benzene_bonds, 1, ["C1C", "C2C"]),
        (benzene_elements, benzene_bonds, 2, ["C1C2C", "C2C1C"]),
        (toluene_elements, toluene_bonds, 1, ["C1C", "C2C"]),
        (toluene_elements, toluene_bonds, 2, ["C1C2C", "C2C1C", "C1C1C"]),
        (propane_elements, propane_bonds, 1, ["C1C"]),
        (propane_elements, propane_bonds, 2, ["C1C1C"]),
        (cyclo_propane_elements, cyclo_propane_bonds, 1, ["C1C"]),
        (cyclo_propane_elements, cyclo_propane_bonds, 2, ["C1C1C"]),
    ],
)
def test_getAllSmiPaths(elements, bonds, length, expected):
    mol = Mol(elements, bonds)
    all_walks = mol.getAllExhaustivePaths(length)
    all_walks = list(map(mol.atomPathToSmi, all_walks))

    assert sorted(list(set(all_walks))) == sorted(
        list(set(expected))
    ), "Expected All Smi paths not found!"


@pytest.mark.parametrize(
    "elements, bonds, expected_file",
    [
        (benzene_elements, benzene_bonds, mykit_path / "tests" / "benzene_fp.npy"),
        (toluene_elements, toluene_bonds, mykit_path / "tests" / "toluene_fp.npy"),
        (propane_elements, propane_bonds, mykit_path / "tests" / "propane_fp.npy"),
        (
            cyclo_propane_elements,
            cyclo_propane_bonds,
            mykit_path / "tests" / "cyclo_propane_fp.npy",
        ),
    ],
)
def test_fingerprint(elements, bonds, expected_file):
    mol = Mol(elements, bonds)
    fp = mol.fingerprint()

    previous_fp = np.load(expected_file)
    assert np.array_equal(fp, previous_fp), "Fingerprints are different!"


@pytest.mark.parametrize(
    "elementsA, bondsA, elementsB, bondsB, expected",
    [
        (benzene_elements, benzene_bonds, benzene_elements, benzene_bonds, True),
        (toluene_elements, toluene_bonds, benzene_elements, benzene_bonds, False),
        (
            propane_elements,
            propane_bonds,
            cyclo_propane_elements,
            cyclo_propane_bonds,
            False,
        ),
        (
            cyclo_propane_elements,
            cyclo_propane_bonds,
            benzene_elements,
            benzene_bonds,
            False,
        ),
    ],
)
def test_equality(elementsA, bondsA, elementsB, bondsB, expected):
    molA = Mol(elementsA, bondsA)
    molB = Mol(elementsB, bondsB)

    assert (molA == molB) == expected, "Comparison operator failed"


@pytest.mark.parametrize(
    "elementsA, bondsA, elementsB, bondsB, expected",
    [
        (benzene_elements, benzene_bonds, benzene_elements, benzene_bonds, True),
        (toluene_elements, toluene_bonds, benzene_elements, benzene_bonds, True),
        (benzene_elements, benzene_bonds, toluene_elements, toluene_bonds, False),
        (
            propane_elements,
            propane_bonds,
            cyclo_propane_elements,
            cyclo_propane_bonds,
            False,
        ),
        (
            cyclo_propane_elements,
            cyclo_propane_bonds,
            benzene_elements,
            benzene_bonds,
            False,
        ),
    ],
)
def test_hasSubstructMatch(elementsA, bondsA, elementsB, bondsB, expected):
    molA = Mol(elementsA, bondsA)
    molB = Mol(elementsB, bondsB)

    A_has_B = molA.hasSubstructMatch(molB)
    assert A_has_B == expected, "Failed substruct search"
