"""
Provided functions.

"""


from typing import Optional


def parse_sdf(
    filename: str,
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

    try:
        with open(filename) as f:
            data = [x.strip() for x in f.readlines()]
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Could not find {filename}. Please check the path and try again."
        )

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
        bonds = [
            x for x in bonds if elements[x[0] - 1] != "H" and elements[x[1] - 1] != "H"
        ]

        heavy_atoms = [idx for idx, atom in enumerate(elements) if atom != "H"]
        elements = [elements[idx] for idx in heavy_atoms]

        if include_coords:
            coords = [coords[idx] for idx in heavy_atoms]

    return elements, bonds, coords
