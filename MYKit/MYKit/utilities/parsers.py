"""
Provided functions.

"""

from pathlib import Path
from typing import Optional, Union


def parse_sdf(
    filename: [str, Path],
    include_hydrogen: Optional[bool] = False,
    include_coords: bool = True,
) -> tuple[dict, list]:
    """
    Read an sdf file and return the atom and bond info

    Parameters
    ----------
    filename
        The name of the file to analyze
    include_hydrogen
        Controls whether information about hydrogens is returned.

    Returns
    -------
    names_and_elements
        A dictionary where the keys are the atom names and the values are the
        elements.
    bonds
        A list of tuples where the first two numbers represent the atom indices
        and the third represents the bond order.
    """

    filename = Path(filename)
    if not filename.exists():
        raise FileNotFoundError(
            f"Could not find {filename}. Please check the path and try again."
        )

    with open(filename, "r") as f:
        data = [x.strip() for x in f.readlines()]

    # read the number of atoms and bonds from the SDF file
    num_atoms = int(data[3].split()[0])
    num_bonds = int(data[3].split()[1])

    # define the end of the atoms and bonds sections
    atom_end = 4 + num_atoms
    bond_end = 4 + num_atoms + num_bonds

    # grab atom and bonds sections from file.
    atoms = data[4:atom_end]
    bonds = data[atom_end:bond_end]

    # check for consistency -
    # does what we retrieved match what is in the top of the file?
    if num_atoms != len(atoms):
        raise ValueError(
            f"Error reading {filename}."
            + "The number of atoms does not match the number of atoms in the file."
        )
    if num_bonds != len(bonds):
        raise ValueError(
            f"Error reading {filename}."
            + "The number of bonds does not match the number of bonds in the file."
        )

    # extract the elements from the atoms section
    elements = [x.split()[3] for x in atoms]
    if include_coords:
        coords = [tuple(map(float, x.split()[0:3])) for x in atoms]
    else:
        coords = None

    # Get list of bonds - format [ (atom1, atom2, bond_order) ...]
    bonds = [
        (int(x.split()[0]) - 1, int(x.split()[1]) - 1, int(x.split()[2])) for x in bonds
    ]

    # remove hydrogens.
    if not include_hydrogen:
        index_remap = {}
        idx = 0
        for i, element in enumerate(elements):
            if element != "H":
                index_remap[i] = idx
                idx += 1

        elements = [element for i, element in enumerate(elements) if i in index_remap]
        bonds = [
            (index_remap[b[0]], index_remap[b[1]], b[2])
            for b in bonds
            if b[0] in index_remap and b[1] in index_remap
        ]

        if include_coords:
            coords = [coord for i, coord in enumerate(coords) if i in index_remap]

    return elements, bonds, coords
