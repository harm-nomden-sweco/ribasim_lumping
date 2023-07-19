import geopandas as gpd
import networkx as nx
from typing import List, Dict, Any, Union, Optional, Tuple


def create_graph_based_on_nodes_edges(
    nodes: gpd.GeoDataFrame, edges: gpd.GeoDataFrame
) -> nx.DiGraph:
    """create networkx graph based on geographic nodes and edges. 
    TODO: maybe a faster implementation possible"""
    graph = nx.DiGraph()
    for i, node in nodes.iterrows():
        graph.add_node(node.mesh1d_nNodes, pos=(node.mesh1d_node_x, node.mesh1d_node_y))
    for i, edge in edges.iterrows():
        graph.add_edge(edge.start_node_no, edge.end_node_no)
    return graph


def split_graph_based_on_node_id(
    graph: nx.DiGraph, split_node_ids: List[int]
) -> nx.DiGraph:
    """split networkx graph at split_node_ids"""
    for split_node_id in split_node_ids:
        split_node_pos = graph.nodes[split_node_id]["pos"]
        split_edges = [e for e in list(graph.edges) if split_node_id in e]
        graph.remove_edges_from(split_edges)
        graph.remove_node(split_node_id)

        for i_edge, new_edge in enumerate(split_edges):
            new_node_id = 999_000_000_000 + split_node_id * 1_000 + i_edge
            graph.add_node(new_node_id, pos=split_node_pos)
            new_edge_adj = [e if e != split_node_id else new_node_id for e in new_edge]
            graph.add_edge(new_edge_adj[0], new_edge_adj[1])
    return graph


def add_basin_code_from_network_to_nodes_and_edges(
    graph: nx.DiGraph,
    nodes: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
    split_node_ids: List[int],
) -> Tuple[gpd.GeoDataFrame]:
    """add basin (subgraph) code to nodes and edges"""
    subgraphs = list(nx.weakly_connected_components(graph))
    nodes["basin"] = 0
    edges["basin"] = 0
    for i, subgraph in enumerate(subgraphs):
        node_ids = list(subgraph) + split_node_ids
        edges.loc[
            edges["start_node_no"].isin(node_ids) & edges["end_node_no"].isin(node_ids),
            "basin",
        ] = i
        nodes.loc[nodes["mesh1d_nNodes"].isin(list(subgraph)), "basin"] = i
    return nodes, edges


def add_basin_code_from_edges_to_areas_create_basin(
    edges: gpd.GeoDataFrame, areas: gpd.GeoDataFrame
) -> Tuple[gpd.GeoDataFrame]:
    """find areas with spatial join on edges. add subgraph code to areas
    and combine all areas with certain subgraph code into one basin"""
    edges_sel = edges[~edges["basin"].isna()]
    areas = areas.sjoin(edges_sel[["basin", "geometry"]]).drop(columns=["index_right"])
    areas = areas.reset_index().rename(columns={"index": "area"})
    areas = (
        areas.groupby(by=[areas["geometry"].to_wkt(), "area", "basin"], as_index=False)
        .size()
        .rename(columns={"level_0": "geometry"})
    )
    areas = gpd.GeoDataFrame(
        areas[["area", "basin", "size"]],
        geometry=gpd.GeoSeries.from_wkt(areas["geometry"]),
    )
    areas = areas.sort_values(
        by=["area", "size"], ascending=[True, False]
    ).drop_duplicates(subset=["area"], keep="first")
    basin_areas = areas.dissolve(by="basin").reset_index()
    return areas, basin_areas


def create_basins_based_on_split_node_ids(
    nodes: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
    split_node_ids: List[int],
    areas: gpd.GeoDataFrame,
) -> Tuple[gpd.GeoDataFrame]:
    """create basins (large polygons) based on nodes, edges, split_node_ids and
    areas (discharge units). call all functions"""
    network_graph = create_graph_based_on_nodes_edges(nodes, edges)
    network_graph = split_graph_based_on_node_id(network_graph, split_node_ids)

    nodes, edges = add_basin_code_from_network_to_nodes_and_edges(
        network_graph, nodes, edges, split_node_ids
    )
    areas, basin_areas = add_basin_code_from_edges_to_areas_create_basin(edges, areas)
    return basin_areas, areas, nodes, edges


def create_additional_basins_for_main_channels(
    nodes: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
    main_channels: gpd.GeoDataFrame,
    basin: gpd.GeoDataFrame,
):
    """for input main_channels (Linstrings), find connected edges/nodes and define whether"""
    raise ValueError('not yet implemented')
