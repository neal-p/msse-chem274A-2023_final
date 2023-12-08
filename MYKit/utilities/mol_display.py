import networkx as nx
from matplotlib import pyplot as plt
import numpy as np
from plotly import graph_objects as go

##############################
#   STYLING MOLECULE IMAGES  #
##############################

ATOM_COLORS = {
    "C": "#D3D3D3",
    "N": "#0000FF",
    "O": "#FF0000",
    "P": "#FFA500",
    "H": "#FFFFFF",
    "S": "#FFFF00",
    "unknown": "#D3D3D3",
}


def get_atom_color(element):
    if element in ATOM_COLORS:
        return ATOM_COLORS[element]
    else:
        return ATOM_COLORS["unknown"]


ATOM_SIZES = {
    # sizes from https://en.wikipedia.org/wiki/Atomic_radii_of_the_elements_(data_page)
    # these are just used to get relative sizes
    # atoms are scaled by image size when displayed
    "C": 70.0,
    "N": 65.0,
    "O": 60.0,
    "P": 100.0,
    "H": 25.0,
    "S": 100.0,
    "unknown": 70.0,
}


def get_atom_size(element):
    if element in ATOM_SIZES:
        return ATOM_SIZES[element]
    else:
        return ATOM_SIZES["unknown"]


BOND_WEIGHT = {
    1: 1.0,
    2: 5.0,
    3: 10.0,
    4: 2.5,
}


def get_bond_style(bond_order):
    return BOND_WEIGHT[bond_order]


def project_xy(graph):
    return {n: (v["x"], v["y"]) for n, v in graph.nodes(data=True)}


def project_xz(graph):
    return {n: (v["x"], v["z"]) for n, v in graph.nodes(data=True)}


def project_yz(graph):
    return {n: (v["y"], v["z"]) for n, v in graph.nodes(data=True)}


def use2D(graph):
    return {n: (v["x"], v["y"]) for n, v in graph.nodes(data=True)}


def use3D(graph):
    return {n: (v["x"], v["y"], v["z"]) for n, v in graph.nodes(data=True)}


LAYOUT_CHOICES = {
    "circular": nx.circular_layout,
    "kamada_kawai": nx.kamada_kawai_layout,
    "random": nx.random_layout,
    "shell": nx.shell_layout,
    "spring": nx.spring_layout,
    "project_xy": project_xy,
    "project_xz": project_xz,
    "project_yz": project_yz,
    "use2D": use2D,
    "use3D": use3D,
}


###############################
#    Main display functions   #
###############################


def DisplayMol(mol, size=(5, 5), layout="infer", file=None):
    if layout == "infer":
        if mol.coords is not None:
            layout = "use2D"
        else:
            layout = "kamada_kawai"

    elif layout not in LAYOUT_CHOICES:
        raise ValueError(f"layout must be one of: {LAYOUT_CHOICES}")

    fig, ax = plt.subplots(figsize=size)

    # Get atom colors, sizes, and bond thicknesses
    elements = [mol.graph.nodes[a]["element"] for a in mol.graph.nodes]
    colors = list(map(get_atom_color, elements))
    sizes = np.array(list(map(get_atom_size, elements)))
    sizes /= np.min(sizes)
    sizes *= 50 * (np.mean(size))
    edge_styles = np.array(
        list(
            map(
                get_bond_style, [e[2]["bond_order"] for e in mol.graph.edges(data=True)]
            )
        )
    )

    pos = LAYOUT_CHOICES[layout](mol.graph)
    nx.draw_networkx_nodes(
        mol.graph, pos, node_color=colors, node_size=sizes, edgecolors="k", ax=ax
    )
    nx.draw_networkx_labels(
        mol.graph,
        pos,
        labels={n: d["element"] for n, d in mol.graph.nodes(data=True)},
        ax=ax,
    )
    nx.draw_networkx_edges(mol.graph, pos, width=edge_styles, ax=ax)

    if file is not None:
        plt.savefig(file)
    else:
        plt.show()


def DisplayMol3D(mol, size=(5, 5), file=None):
    layout = go.Layout(showlegend=False)
    fig = go.Figure(layout=layout)

    # Get atom colors, sizes, and bond thicknesses
    elements = [mol.graph.nodes[a]["element"] for a in mol.graph.nodes]
    colors = list(map(get_atom_color, elements))
    sizes = np.array(list(map(get_atom_size, elements)))
    sizes /= np.min(sizes)
    sizes *= 5 * (np.mean(size))
    edge_styles = np.array(
        list(
            map(
                get_bond_style, [e[2]["bond_order"] for e in mol.graph.edges(data=True)]
            )
        )
    )

    pos = LAYOUT_CHOICES["use3D"](mol.graph)

    node_xyz = np.array([pos[n] for n in mol.graph.nodes()])

    fig.add_trace(
        go.Scatter3d(
            x=node_xyz.T[0],
            y=node_xyz.T[1],
            z=node_xyz.T[2],
            mode="markers",
            marker=dict(color=colors, size=sizes, line=dict(color="black")),
        )
    )

    for idx, (start, end) in enumerate(mol.graph.edges()):
        fig.add_trace(
            go.Scatter3d(
                x=[pos[start][0], pos[end][0]],
                y=[pos[start][1], pos[end][1]],
                z=[pos[start][2], pos[end][2]],
                line=dict(color="black", width=edge_styles[idx]),
                mode="lines",
            )
        )

    invisible_scale = go.Scatter3d(
        name="",
        visible=True,
        showlegend=False,
        opacity=0,
        hoverinfo="none",
        x=[np.min(node_xyz[:, 0]), np.max(node_xyz[:, 0])],
        y=[np.min(node_xyz[:, 1]), np.max(node_xyz[:, 1])],
        z=[np.min(node_xyz[:, 2]), np.max(node_xyz[:, 2])],
    )

    fig.add_trace(invisible_scale)

    fig.update_layout(
        scene=dict(
            camera=dict(dict(projection=dict(type="orthographic"))),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
        )
    )

    if file is not None:
        fig.write_html(file)

    else:
        fig.show()
