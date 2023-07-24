import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union, Optional, Tuple
from shapely.geometry import Polygon
from shapely.ops import polylabel


def create_graph_based_on_nodes_edges(
    nodes: gpd.GeoDataFrame, edges: gpd.GeoDataFrame
) -> nx.DiGraph:
    """create networkx graph based on geographic nodes and edges. 
    TODO: maybe a faster implementation possible"""
    graph = nx.DiGraph()
    if nodes is not None:
        for i, node in nodes.iterrows():
            graph.add_node(node.mesh1d_nNodes, pos=(node.mesh1d_node_x, node.mesh1d_node_y))
    if edges is not None:
        for i, edge in edges.iterrows():
            graph.add_edge(edge.start_node_no, edge.end_node_no)
    return graph


def split_graph_based_on_node_id(
    graph: nx.DiGraph, split_nodes: gpd.GeoDataFrame
) -> nx.DiGraph:
    """split networkx graph at split_node_ids"""
    for split_node_id in split_nodes.mesh1d_nNodes.values:
        if split_node_id not in graph:
            continue
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
    split_nodes: List[int],
) -> Tuple[gpd.GeoDataFrame]:
    """add basin (subgraph) code to nodes and edges"""
    subgraphs = list(nx.weakly_connected_components(graph))
    if nodes is None or edges is None:
        return None, None
    nodes["basin"] = -1
    edges["basin"] = -1
    for i, subgraph in enumerate(subgraphs):
        node_ids = list(subgraph) + list(split_nodes.mesh1d_nNodes.values)
        edges.loc[
            edges["start_node_no"].isin(node_ids) & edges["end_node_no"].isin(node_ids),
            "basin",
        ] = i
        nodes.loc[nodes["mesh1d_nNodes"].isin(list(subgraph)), "basin"] = i
    return nodes, edges


def add_basin_code_from_edges_to_areas_and_create_basin(
    edges: gpd.GeoDataFrame, areas: gpd.GeoDataFrame
) -> Tuple[gpd.GeoDataFrame]:
    """find areas with spatial join on edges. add subgraph code to areas
    and combine all areas with certain subgraph code into one basin"""
    if edges is None:
        if areas is None:
            return None, None
        areas['basin'] = np.nan
        return areas, None
    edges_sel = edges[~edges["basin"].isna()]
    areas = areas.sjoin(edges_sel[["basin", "geometry"]]).drop(columns=["index_right"])
    areas = areas.reset_index().rename(columns={"index": "area"})
    areas = (
        areas.groupby(by=[areas["geometry"].to_wkt(), "area", "basin"], as_index=False)
        .size()
        .rename(columns={"level_0": "geometry", "size": "no_nodes"})
    )
    areas = gpd.GeoDataFrame(
        areas[["area", "basin", "no_nodes"]],
        geometry=gpd.GeoSeries.from_wkt(areas["geometry"]),
    )
    areas = areas.sort_values(
        by=["area", "no_nodes"], ascending=[True, False]
    ).drop_duplicates(subset=["area"], keep="first")
    basin_areas = areas.dissolve(by="basin").reset_index().drop(columns=['area'])
    return areas, basin_areas


def create_basins_based_on_basin_areas_or_nodes(basin_areas, nodes):
    if basin_areas is None:
        basins = None
    else:
        basins = nodes[nodes.basin!=-1].groupby(by='basin').agg({
            'mesh1d_nNodes': 'size', 
            'mesh1d_node_x': 'mean', 
            'mesh1d_node_y': 'mean'
        }).reset_index().rename(columns={'mesh1d_nNodes': 'no_nodes'})
        basins = gpd.GeoDataFrame(
            data=basins[['basin', 'no_nodes']],
            geometry=gpd.points_from_xy(basins.mesh1d_node_x, basins.mesh1d_node_y),
            crs=nodes.crs
        )

        basins_pnt_gdf = basin_areas.copy()
        basins_pnt_gdf.geometry = [polylabel(g) for g in basins_pnt_gdf.geometry]
        basins_pnt_gdf = basins_pnt_gdf.set_crs(nodes.crs)
        basins = gpd.GeoDataFrame(
            pd.concat([basins_pnt_gdf, basins[~basins.basin.isin(basins_pnt_gdf.basin)]]),
            crs=nodes.crs
        )
    return basins


def check_if_split_node_is_used(split_nodes, nodes, edges):
    split_node_ids = list(split_nodes.mesh1d_nNodes.values)
    split_nodes_not_used = []
    for split_node_id in split_node_ids:
        end_nodes = list(edges[edges.start_node_no == split_node_id].end_node_no.values)
        start_nodes = list(edges[edges.end_node_no == split_node_id].start_node_no.values)
        neighbours = nodes[nodes.mesh1d_nNodes.isin(end_nodes + start_nodes)]
        if len(neighbours.basin.unique()) == 1:
            split_nodes_not_used.append(split_node_id)
    split_nodes['status'] = True
    split_nodes.loc[split_nodes[split_nodes.mesh1d_nNodes.isin(split_nodes_not_used)].index, 'status'] = False
    return split_nodes


def create_basins_using_split_nodes(
    nodes: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
    split_nodes: gpd.GeoDataFrame,
    areas: gpd.GeoDataFrame,
) -> Tuple[gpd.GeoDataFrame]:
    """create basins (large polygons) based on nodes, edges, split_nodes and
    areas (discharge units). call all functions"""
    if areas is not None:
        if areas.crs is None:
            areas = areas[['geometry']].set_crs(28992)
        else:
            areas = areas[['geometry']].to_crs(28992)
    network_graph = create_graph_based_on_nodes_edges(nodes, edges)
    network_graph = split_graph_based_on_node_id(network_graph, split_nodes)

    nodes, edges = add_basin_code_from_network_to_nodes_and_edges(
        network_graph, nodes, edges, split_nodes
    )
    split_nodes = check_if_split_node_is_used(split_nodes, nodes, edges)
    areas, basin_areas = add_basin_code_from_edges_to_areas_and_create_basin(edges, areas)
    basins = create_basins_based_on_basin_areas_or_nodes(basin_areas, nodes)

    return basin_areas, basins, areas, nodes, edges, split_nodes


def create_additional_basins_for_main_channels(
    nodes: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
    main_channels: gpd.GeoDataFrame,
    basin: gpd.GeoDataFrame,
):
    """for input main_channels (Linstrings), find connected edges/nodes and define whether"""
    raise ValueError('not yet implemented')
