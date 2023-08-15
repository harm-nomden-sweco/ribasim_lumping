import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union, Optional, Tuple
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import polylabel
from .general_functions import create_objects_gdf


def create_graph_based_on_nodes_edges(
    nodes: gpd.GeoDataFrame, 
    edges: gpd.GeoDataFrame
) -> nx.DiGraph:
    """create networkx graph based on geographic nodes and edges. 
    TODO: maybe a faster implementation possible"""
    print(" - create network graph from nodes/edges")
    graph = nx.DiGraph()
    if nodes is not None:
        for i, node in nodes.iterrows():
            graph.add_node(node.mesh1d_nNodes, pos=(node.geometry.x, node.geometry.y))
    if edges is not None:
        for i, edge in edges.iterrows():
            graph.add_edge(edge.start_node_no, edge.end_node_no)
    return graph


def split_graph_based_on_split_nodes(
    graph: nx.DiGraph, 
    split_nodes: gpd.GeoDataFrame, 
    edges_gdf: gpd.GeoDataFrame
) -> nx.DiGraph:
    """split networkx graph at split_edge or split_node"""
    print(" - split network graph at split locations")

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
    print(" - define numbers Ribasim-Basins and join edges/nodes")
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
    """check whether split_nodes are used, split_nodes and split_edges"""
    print(" - check whether each split location results in a split")
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


def create_basin_areas_based_on_drainage_areas(
    edges: gpd.GeoDataFrame, areas: gpd.GeoDataFrame
) -> Tuple[gpd.GeoDataFrame]:
    """find areas with spatial join on edges. add subgraph code to areas
    and combine all areas with certain subgraph code into one basin"""
    print(" - define for each Ribasim-Basin the related basin area")
    if areas is None:
        return None, None
    else:
        areas = areas[['geometry']].copy()
    if edges is None:
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


def create_basins_based_on_subgraphs_and_nodes(graph, nodes):
    """create basin nodes based on basin_areas or nodes"""
    print(" - create final locations Ribasim-Basins")
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
    basins = tmp.groupby(by='basin').first().reset_index().set_crs(nodes.crs)
    return basins


def create_basin_connections(
        split_nodes: gpd.GeoDataFrame, 
        edges: gpd.GeoDataFrame, 
        nodes: gpd.GeoDataFrame, 
        basins: gpd.GeoDataFrame,
        crs: int = 28992,
    ) -> gpd.GeoDataFrame:
    """create basin connections"""
    print(" - create Ribasim-Edges between Basins and split locations")

    conn = (split_nodes[['mesh1d_nNodes','geometry', 'split_type', 'mesh1d_node_id', 'mesh1d_nEdges']]
            .rename(columns={"geometry":"geom_split_node"}))
    # check if split_node is used (split_type)
    conn = conn[conn['split_type']!='no_split']

    # use different approach for: (1) splitnodes that are structures and on an edge and (2) splitnodes that are original d-hydro nodes

    # (1) splitnodes that are structures and on an edge
    conn_struct = conn.loc[conn['mesh1d_nNodes']==-1]
    # merge with edge to find us and ds nodes
    conn_struct = conn_struct.merge(
        edges[['start_node_no', 'end_node_no','mesh1d_nEdges']],
        left_on='mesh1d_nEdges', 
        right_on='mesh1d_nEdges'
    )
    # merge with node to find us and ds basin
    conn_struct_us = conn_struct.merge(
        nodes,
        left_on='start_node_no', 
        right_on='mesh1d_nNodes',
        suffixes=('','_r'),
    ).drop(columns=['start_node_no','end_node_no','mesh1d_nNodes_r','mesh1d_node_id_r', 'geometry'])
    conn_struct_ds = conn_struct.merge(
        nodes,
        left_on='end_node_no', 
        right_on='mesh1d_nNodes',
        suffixes=('','_r')
    ).drop(columns=['start_node_no','end_node_no','mesh1d_nNodes_r','mesh1d_node_id_r', 'geometry'])
    
    # (2) splitnodes that are original d-hydro nodes
    # merge splitnodes add connected edges
    conn_nodes = conn.loc[conn['mesh1d_nNodes']!=-1].drop(columns=['mesh1d_nEdges'])

    conn_nodes_ds = conn_nodes.merge(
        edges[['basin', 'start_node_no', 'end_node_no','mesh1d_nEdges']],
        left_on='mesh1d_nNodes', 
        right_on='start_node_no'
    )
    conn_nodes_us = conn_nodes.merge(
        edges[['basin', 'start_node_no','end_node_no','mesh1d_nEdges']],
        left_on='mesh1d_nNodes', 
        right_on='end_node_no'
    )

    # Combine (1) en (2)
    conn_ds = pd.concat([conn_nodes_ds, conn_struct_ds])
    conn_us = pd.concat([conn_nodes_us, conn_struct_us])
    # merge splitnodes with basin DOWNSTREAM
    conn_ds = conn_ds.merge(
        basins[['basin', 'geometry']], 
        left_on='basin', 
        right_on='basin'
    ).rename(columns={"geometry":"geom_basin"})
    conn_ds['side'] = 'downstream'
    # merge splitnodes with basin UPSTREAM
    conn_us = conn_us.merge(
        basins[['basin', 'geometry']], 
        left_on='basin', 
        right_on='basin'
    ).rename(columns={"geometry": "geom_basin"})
    conn_us['side'] = 'upstream'
    # merge up- and downstream
    conn_gdf = conn_us.merge(
        conn_ds, 
        left_on='mesh1d_node_id',
        right_on='mesh1d_node_id',
        suffixes=('_out','_in')
    )
    conn_gdf = conn_gdf[[
        'split_type_out', 'mesh1d_node_id', 'basin_out', 'basin_in',
        'geom_basin_out', 'geom_split_node_out', 'geom_basin_in'
    ]]
    # draw connection line via split node
    conn_gdf['geometry'] = conn_gdf.apply(
        lambda row: LineString([row['geom_basin_out'], row['geom_split_node_out'], row['geom_basin_in']]), 
        axis=1
    )
    conn_gdf = gpd.GeoDataFrame(conn_gdf, geometry='geometry', crs=crs)
    basin_connections_gdf = conn_gdf.drop(columns=['geom_basin_in','geom_basin_out','geom_split_node_out'])
    return basin_connections_gdf


def create_boundary_connections(
        boundaries: gpd.GeoDataFrame, 
        edges: gpd.GeoDataFrame, 
        basins: gpd.GeoDataFrame,
    ) -> gpd.GeoDataFrame:
    """create boundary-basin connections"""
    print(" - create Ribasim-Edges between Boundary and Basin")
    if boundaries is None or edges is None or basins is None:
        return None

    # merge boundaries with edges 
    boundary_conn = boundaries.rename(columns={"geometry":"geometry_boundary"})
    boundary_conn_us = boundary_conn.merge(
        edges[['start_node_no', 'end_node_no','mesh1d_nEdges', 'basin']], 
        left_on='mesh1d_nNodes', 
        right_on='start_node_no'
    )
    boundary_conn_ds = boundary_conn.merge(
        edges[['start_node_no','end_node_no','mesh1d_nEdges', 'basin']], 
        left_on='mesh1d_nNodes', 
        right_on='end_node_no'
    )

    # merge with basins for geometry
    # downstream
    boundary_conn_ds = boundary_conn_ds.merge(
        basins[['basin', 'geometry']], 
        left_on='basin', 
        right_on='basin'
    ).rename(columns={"geometry":"geometry_basin"})
    boundary_conn_ds['boundary_location'] = 'downstream'
    boundary_conn_ds['geometry'] = boundary_conn_ds.apply(
        lambda row: LineString([row['geometry_basin'], row['geometry_boundary']]), 
        axis=1
    )
    boundary_conn_ds = gpd.GeoDataFrame(boundary_conn_ds, geometry='geometry', crs=28992)

    # upstream
    boundary_conn_us = boundary_conn_us.merge(
        basins[['basin', 'geometry']], 
        left_on='basin', 
        right_on='basin',
    ).rename(columns={"geometry":"geometry_basin"})
    boundary_conn_us['boundary_location'] = 'upstream'
    boundary_conn_us['geometry'] = boundary_conn_us.apply(
        lambda row: LineString([row['geometry_boundary'], row['geometry_basin']]), 
        axis=1
    )
    boundary_conn_us = gpd.GeoDataFrame(boundary_conn_us, geometry='geometry', crs=28992)

    # concat us and ds
    boundary_conn_gdf = pd.concat([boundary_conn_us, boundary_conn_ds])
    # basin_connections = ribasim_edges_gdf.copy()

    boundary_conn_gdf = boundary_conn_gdf.drop(columns=['geometry_boundary','geometry_basin'])
    boundary_conn_gdf.insert(0, 'boundary_conn_id', range(len(boundary_conn_gdf)))
    boundary_conn_gdf = boundary_conn_gdf[
        ['boundary_conn_id', 'boundary_id', 'basin', 'boundary_location', 'geometry']
    ]
    return boundary_conn_gdf


def create_basins_and_connections_using_split_nodes(
    nodes: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
    split_nodes: gpd.GeoDataFrame,
    areas: gpd.GeoDataFrame,
    boundaries: gpd.GeoDataFrame,
    crs: int = 28992
) -> Tuple[gpd.GeoDataFrame]:
    """create basins (nodes) and basin_areas (large polygons) and connections (edges) 
    based on nodes, edges, split_nodes and areas (discharge units). 
    This function calls all other functions"""
    print("Create basins using split nodes:")
    network_graph = None
    basin_areas = None
    basins = None
    network_graph = create_graph_based_on_nodes_edges(nodes, edges)
    network_graph = split_graph_based_on_split_nodes(network_graph, split_nodes, edges)
    nodes, edges = add_basin_code_from_network_to_nodes_and_edges(
        network_graph, nodes, edges, split_nodes
    )
    split_nodes = check_if_split_node_is_used(split_nodes, nodes, edges)
    basins = create_basins_based_on_subgraphs_and_nodes(network_graph, nodes)
    areas, basin_areas = create_basin_areas_based_on_drainage_areas(edges, areas)
    boundary_conn = create_boundary_connections(boundaries, edges, basins)
    basin_connections = create_basin_connections(split_nodes, edges, nodes, basins, crs)

    return basin_areas, basins, areas, nodes, edges, split_nodes, \
        network_graph, basin_connections, boundary_conn

