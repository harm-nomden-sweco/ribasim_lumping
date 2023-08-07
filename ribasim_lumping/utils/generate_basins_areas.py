import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union, Optional, Tuple
from shapely.geometry import Polygon, Point
from shapely.ops import polylabel
from .general_functions import create_objects_gdf


def create_graph_based_on_nodes_edges(
    nodes: gpd.GeoDataFrame, edges: gpd.GeoDataFrame
) -> nx.DiGraph:
    """create networkx graph based on geographic nodes and edges. 
    TODO: maybe a faster implementation possible"""
    print(" - create network graph")
    graph = nx.DiGraph()
    if nodes is not None:
        for i, node in nodes.iterrows():
            graph.add_node(node.mesh1d_nNodes, pos=(node.geometry.x, node.geometry.y))
    if edges is not None:
        for i, edge in edges.iterrows():
            graph.add_edge(edge.start_node_no, edge.end_node_no)
    return graph


def split_graph_based_on_split_nodes(
    graph: nx.DiGraph, split_nodes: gpd.GeoDataFrame, edges_gdf: gpd.GeoDataFrame
) -> nx.DiGraph:
    """split networkx graph at split_edge or split_node"""
    print(" - split network graph")

    # split on edge: delete edge, create 2 nodes, create 2 edges
    split_nodes_edges = split_nodes[split_nodes.mesh1d_nEdges!=-1].copy()
    split_edges = edges_gdf[edges_gdf.mesh1d_nEdges.isin(split_nodes_edges.mesh1d_nEdges.values)].copy()
    split_edges = split_edges[['start_node_no', 'end_node_no']].to_dict('tight')['data']
    split_edges = [coor for coor in split_edges if coor in graph.edges]

    graph.remove_edges_from(split_edges)

    split_nodes_edges['new_node_no1'] = 998_000_000_000 + split_nodes_edges.mesh1d_nEdges * 1_000 + 1
    split_nodes_edges['new_node_no2'] = 998_000_000_000 + split_nodes_edges.mesh1d_nEdges * 1_000 + 2
    split_nodes_edges['new_node_pos'] = split_nodes_edges.geometry.apply(lambda x: (x.x, x.y))

    split_nodes_edges['upstream_node_no'] = [e[0] for e in split_edges]
    split_nodes_edges['downstream_node_no'] = [e[1] for e in split_edges]

    for i_edge, new in split_nodes_edges.iterrows():
        graph.add_node(new.new_node_no1, pos=new.new_node_pos)
        graph.add_node(new.new_node_no2, pos=new.new_node_pos)
        graph.add_edge(new.upstream_node_no, new.new_node_no1)
        graph.add_edge(new.new_node_no2, new.downstream_node_no)

    # split_node: delete node and delete x edges, create x nodes, create x edges
    split_nodes_nodes = split_nodes[split_nodes.mesh1d_nNodes!=-1]
    for split_node_id in split_nodes_nodes.mesh1d_nNodes.values:
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
    print(" - define basins based on sub-networks")
    subgraphs = list(nx.weakly_connected_components(graph))
    if nodes is None or edges is None:
        return None, None
    nodes["basin"] = -1
    edges["basin"] = -1
    for i, subgraph in enumerate(subgraphs):
        node_ids = list(subgraph) + list(split_nodes.mesh1d_nNodes.values)
        edges.loc[
            edges["start_node_no"].isin(node_ids) & 
            edges["end_node_no"].isin(node_ids),
            "basin",
        ] = i
        nodes.loc[nodes["mesh1d_nNodes"].isin(list(subgraph)), "basin"] = i
    return nodes, edges


def check_if_split_node_is_used(split_nodes, nodes, edges):
    """check if split_nodes are used, split_nodes and split_edges"""
    print(" - check if split locations are used")
    split_nodes['status'] = True

    # check if edges connected to split_nodes have the same basin code
    split_node_ids = [v for v in split_nodes.mesh1d_nNodes.values if v != -1]
    split_nodes_not_used = []
    for split_node_id in split_node_ids:
        end_nodes = list(edges[edges.start_node_no == split_node_id].end_node_no.values)
        start_nodes = list(edges[edges.end_node_no == split_node_id].start_node_no.values)
        neighbours = nodes[nodes.mesh1d_nNodes.isin(end_nodes + start_nodes)]
        if len(neighbours.basin.unique()) == 1:
            split_nodes_not_used.append(split_node_id)
    split_nodes.loc[split_nodes[split_nodes.mesh1d_nNodes.isin(split_nodes_not_used)].index, 'status'] = False

    # check if nodes connected to split_edge have the same basin code
    split_edge_ids = [v for v in split_nodes.mesh1d_nEdges.values if v != -1]
    split_edges_not_used = []
    for split_edge_id in sorted(split_edge_ids):
        end_nodes = list(edges[edges.mesh1d_nEdges == split_edge_id].end_node_no.values)
        start_nodes = list(edges[edges.mesh1d_nEdges == split_edge_id].start_node_no.values)
        neighbours = nodes[nodes.mesh1d_nNodes.isin(end_nodes + start_nodes)]
        if len(neighbours.basin.unique()) == 1:
            split_edges_not_used.append(split_edge_id)
    split_nodes.loc[split_nodes[split_nodes.mesh1d_nEdges.isin(split_edges_not_used)].index, 'status'] = False

    split_nodes['object_type'] = split_nodes['object_type'].fillna('manual')
    split_nodes['split_type'] = split_nodes['object_type']
    split_nodes.loc[~split_nodes.status, 'split_type'] = 'no_split'
    return split_nodes


def add_basin_code_from_edges_to_areas(
    edges: gpd.GeoDataFrame, areas: gpd.GeoDataFrame
) -> Tuple[gpd.GeoDataFrame]:
    """find areas with spatial join on edges. add subgraph code to areas
    and combine all areas with certain subgraph code into one basin"""
    print(" - define basin areas")
    if edges is None:
        if areas is None:
            return None, None
        areas['basin'] = np.nan
        return areas, None
    edges_sel = edges[edges["basin"]!=-1].copy()
    edges_sel['edge_length'] = edges_sel.geometry.length
    areas['area'] = areas.index
    areas_orig = areas.copy()
    areas = areas.sjoin(edges_sel[["basin", "edge_length", "geometry"]])
    areas = areas.drop(columns=["index_right"]).reset_index(drop=True)
    areas = (
        areas.groupby(by=["area", "basin"], as_index=False)
        .agg({'edge_length': 'sum'})
    )
    areas = areas.sort_values(
        by=["area", "edge_length"], ascending=[True, False]
    ).drop_duplicates(subset=["area"], keep="first")
    areas = (areas[["area", "basin", "edge_length"]]
             .sort_values(by='area')
             .merge(areas_orig, how='outer', left_on='area', right_on='area'))
    areas = gpd.GeoDataFrame(areas, geometry='geometry', crs=edges.crs)
    areas = areas.sort_values(by='area')
    basin_areas = areas.dissolve(by="basin").reset_index().drop(columns=['area'])
    return areas, basin_areas


def create_basins_based_on_basin_areas_or_edges(graph, nodes):
    """create basin nodes based on basin_areas or nodes"""
    print(" - create final basins")
    connected_components = list(nx.weakly_connected_components(graph))
    centralities = {}

    for i, component in enumerate(connected_components):
        subgraph = graph.subgraph(component).to_undirected()
        centrality_subgraph = nx.closeness_centrality(subgraph)
        centralities.update({node: centrality for node, centrality in centrality_subgraph.items()})

    centralities = pd.DataFrame(dict(
        node_id=list(centralities.keys()),
        centrality=list(centralities.values())
    ))
    centralities = centralities[centralities['node_id'] < 900_000_000_000]
    tmp = nodes.merge(centralities, how='outer', left_on='mesh1d_nNodes', right_on='node_id')
    tmp = tmp[tmp['basin']!=-1].sort_values(by=['basin', 'centrality'], ascending=[True, False])
    basins = tmp.groupby(by='basin').first().reset_index()
    return basins


def create_basins_using_split_nodes(
    nodes: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
    split_nodes: gpd.GeoDataFrame,
    areas: gpd.GeoDataFrame,
    crs: int = 28992
) -> Tuple[gpd.GeoDataFrame]:
    """create basins (large polygons) based on nodes, edges, split_nodes and
    areas (discharge units). call all functions"""
    print("Create basins using split nodes:")
    network_graph = None
    basin_areas = None
    basins = None
    if areas is not None:
        if areas.crs is None:
            areas = areas[['geometry']].set_crs(crs)
        else:
            areas = areas[['geometry']].to_crs(crs)
    network_graph = create_graph_based_on_nodes_edges(nodes, edges)
    network_graph = split_graph_based_on_split_nodes(network_graph, split_nodes, edges)
    nodes, edges = add_basin_code_from_network_to_nodes_and_edges(
        network_graph, nodes, edges, split_nodes
    )
    split_nodes = check_if_split_node_is_used(split_nodes, nodes, edges)
    areas, basin_areas = add_basin_code_from_edges_to_areas(edges, areas)
    basins = create_basins_based_on_basin_areas_or_edges(network_graph, nodes)
    return basin_areas, basins, areas, nodes, edges, split_nodes, network_graph

