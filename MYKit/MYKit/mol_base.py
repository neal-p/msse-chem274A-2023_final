import networkx as nx
import hashlib
import pandas as pd
import numpy as np
from typing import List, Tuple, Union, Optional
from pathlib import Path

from MYKit.utilities import parsers
from MYKit.utilities.mol_display import DisplayMol, DisplayMol3D
from MYKit.utilities.ptable import ELEMENT_DICT


class Mol:
    """
    A molecule class to hold the networkx graph and add methods for displaying molecule
    """

    def __init__(
        self,
        atoms: List[str],
        bonds: List[Tuple[int, int, int]],
        attributes={},
        coords: Optional[List[Tuple[float, float, float]]] = None,
    ):
        """
        A class to hold the networkx graph and add methods for displaying molecule

        Parameters
        ----------
        atoms : List[str]
            List of atoms in the molecule
        bonds : List[Tuple[int, int, int]]
            List of tuples containing the index of atoms in a bond and the bond order.
            For example, (0, 1, 1) specifies a single bond between atom0 and atom1.
        attributes : dict, optional
            Additional molecule information. The '_Name' attribute is used when
            printing the molecule information, by default {}
        coords : List[Tuple[float, float, float]], optional
            3D coordinates of the molecule, by default None
        """

        self.attributes = attributes
        self.coords = coords

        # Build from edges to initialize graph
        self.graph = nx.Graph(((a1, a2, {"bond_order": bo}) for a1, a2, bo in bonds))
        nx.set_node_attributes(self.graph, dict(enumerate(atoms)), "element")

        if self.coords is not None:
            # x
            nx.set_node_attributes(
                self.graph, dict(enumerate(map(lambda s: s[0], self.coords))), "x"
            )
            # y
            nx.set_node_attributes(
                self.graph, dict(enumerate(map(lambda s: s[1], self.coords))), "y"
            )
            # z
            nx.set_node_attributes(
                self.graph, dict(enumerate(map(lambda s: s[2], self.coords))), "z"
            )

        # Initialize ring information
        self.rings = self.RingPerception()

    def RingPerception(self) -> List[Tuple[int]]:
        """
        Use networkx to find the rings in the molecule

        Returns
        -------
        List[Tuple[int]]
            A list of tuples representing the atoms in each ring
        """
        try:
            return nx.cycle_basis(self.graph)
        except nx.NetworkXNoCycle:
            return []

    def GetNumRings(self) -> int:
        """
        Return the number of rings in the molecule

        Returns
        -------
        int
            Rings in the molecule
        """
        return len(self.rings)

    def GeRingSizes(self) -> List[int]:
        """
        Return the size of each ring in the molecule

        Returns
        -------
        List[int]
            The size of each ring in the molecule
        """
        return [len(r) for r in self.rings]

    def __str__(self) -> str:
        """
        Print the molecule name, number of atoms,
        number of bonds, and number of rings

        Returns
        -------
        str
            Print the molecule name, number of atoms,
            number of bonds, and number of rings
        """

        if "_Name" in self.attributes:
            name = self.attributes["_Name"]
        else:
            name = ""

        return f"Mol {name} with {len(self.graph.nodes)} atoms, \
            {len(self.graph.edges)} \
            bonds, and {self.GetNumRings()} rings"

    def display(self):
        """
        Basic 2D display
        """
        DisplayMol(self)

    def display3D(self):
        """
        Basic 2D display
        """
        DisplayMol3D(self)

    def _ipython_display_(self):
        """
        Make Mol able to display graph plot when ipython display is called
        """
        DisplayMol(self)

    def plot_to_file(self, file: str, layout: str = "kamada_kawai"):
        """
        Plot the 2D molecule image with networkx

        Parameters
        ----------
        file : str
            File to save molecule image to
        layout : str, optional
            Which networkx graph layout algorithm to use, by default 'kamada_kawai'
        """
        DisplayMol(self, file=file)

    def plot_3D_to_file(self, file: str):
        """
        Plot 3D structure and save interactive figure to HTML

        Parameters
        ----------
        file : str
            File to save the interactive molecule HTML

        Raises
        ------
        ValueError
            Must provide 3D coordinates when requesting to plot 3D
        """
        if self.coords is None:
            raise ValueError("3D coordinates must be provided to request 3D printing")

        DisplayMol3D(self, file=file)

    def __getAtomNeighborhood(self, atom_idx: int, radius: int = 3):
        return nx.ego_graph(self.graph, atom_idx, radius=radius, undirected=True)

    def __getExhaustivePaths(self, start: int, length: int):
        if length == 0:
            return [[start]]

        paths = []
        for neighbor in self.graph.neighbors(start):
            for path in self.__getExhaustivePaths(neighbor, length - 1):
                if len(path) <= 1 or start != path[1]:
                    paths.append([start] + path)

        return paths

    def __getAllExhaustivePaths(self, length: int):
        paths = []
        for node in self.graph.nodes:
            paths.extend(self.__getExhaustivePaths(node, length))

        return paths

    def __atomPathToSmi(self, path):
        output = [self.graph.nodes[path[0]]["element"]]
        for start, end in zip(path[0:-1], path[1:]):
            output.append(str(self.graph.get_edge_data(start, end)["bond_order"]))
            output.append(self.graph.nodes[end]["element"])

        return "".join(output)

    def fingerprint(self, min_path=2, max_path=7, nBits=2048, bitsPerHash=2):
        np_seed_og = np.random.get_state()

        fp = np.zeros(2048)
        all_walks = []
        for length in range(min_path, max_path + 1):
            all_walks.extend(self.__getAllExhaustivePaths(length))

        all_walks = list(map(self.__atomPathToSmi, all_walks))

        for walk in all_walks:
            # need deterministic hash
            h = hashlib.sha256()
            h.update(walk.encode("utf-8"))
            seed = int(h.hexdigest(), 16) % 2**32 - 1

            np.random.seed(seed)
            bits = np.random.randint(0, nBits, bitsPerHash)

            for i in bits:
                fp[i] = 1

        np.random.set_state(np_seed_og)
        return fp

    @property
    def formula(self) -> str:
        """
        Return the molecular formula

        Note, hydrogens are not really handled correctly since we don't have
        a notion of valency in the Mol class :(
        """
        element_counts = {}
        for node, data in self.graph.nodes(data=true):
            element = data["element"]
            if element not in element_counts:
                element_counts[element] = 0
            element_counts[element] += 1

        formula = ""
        if "C" in element_counts:
            formula += "C" + element_counts["C"]
            del element_counts["C"]

        if "N" in element_counts:
            formula += "N" + element_counts["N"]
            del element_counts["N"]

        if "O" in element_counts:
            formula += "O" + element_counts["O"]
            del element_counts["O"]

        for element, count in element_counts.items():
            formula += f"{element }{count}"

        return formula

    def __eq__(self, other, **kwargs) -> bool:
        """
        Equality is defined by:
            Same molecular formula (given by .formula)
            Same connectivity (indirectly given by fingerprint equality)

        kwargs are passed to fingerprint function
        """

        if self.formula != other.formula:
            return False

        return np.array_equal(self.fingerprint(), other.fingerprint())

    def hasSubstructMatch(self, substructure, **kwargs) -> bool:
        """
        Use fingerprint to determine if the given substructure is present in the molecule

        """
        onbits = np.nonzero(substructure.fingerprint(**kwargs))[0]
        fp = self.fingerprint(**kwargs)
        for bit in onbits:
            if fp[bit] != 1:
                return False

        return True


def SDFToMol(file: Union[str, Path], **kwargs) -> Mol:
    """
    Create a Mol from a provided SDF file

    kwargs are passed to read.parse_sdf

    Parameters
    ----------
    file : Union[str, Path]
        The SDF file to parse

    Returns
    -------
    Mol
        The parsed molecule
    """
    file = Path(file)
    atoms, bonds, coords = parsers.parse_sdf(file, **kwargs)

    return Mol(
        atoms,
        bonds,
        attributes={"_Name": file.stem, "SDF_parse_arguments": kwargs},
        coords=coords,
    )
