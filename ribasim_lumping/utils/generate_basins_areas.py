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
            graph.add_node(node.node_no, pos=(node.geometry.x, node.geometry.y))
    if edges is not None:
        for i, edge in edges.iterrows():
            graph.add_edge(edge.from_node, edge.to_node)
    return graph


def split_graph_based_on_split_nodes(
    graph: nx.DiGraph, 
    split_nodes: gpd.GeoDataFrame, 
    edges_gdf: gpd.GeoDataFrame
) -> nx.DiGraph:
    """split networkx graph at split_edge or split_node"""
    print(" - split network graph at split locations")

    # split on edge: delete edge, create 2 nodes, create 2 edges
    split_nodes_edges = split_nodes[split_nodes.edge_no!=-1].copy()
    seen = set()
    dupes = [x for x in split_nodes.edge_no.values if x in seen or seen.add(x)]
    display(split_nodes[split_nodes.edge_no.isin(dupes)])

    split_edges = edges_gdf[edges_gdf.edge_no.isin(split_nodes_edges.edge_no.values)].copy()
    split_edges = split_edges[['from_node', 'to_node']].to_dict('tight')['data']
    
    split_edges = [coor for coor in split_edges if coor in graph.edges]

    graph.remove_edges_from(split_edges)

    split_nodes_edges['new_node_no1'] = 998_000_000_000 + split_nodes_edges.edge_no * 1_000 + 1
    split_nodes_edges['new_node_no2'] = 998_000_000_000 + split_nodes_edges.edge_no * 1_000 + 2
    split_nodes_edges['new_node_pos'] = split_nodes_edges.geometry.apply(lambda x: (x.x, x.y))

    split_nodes_edges['upstream_node_no'] = [e[0] for e in split_edges]
    split_nodes_edges['downstream_node_no'] = [e[1] for e in split_edges]

    for i_edge, new in split_nodes_edges.iterrows():
        graph.add_node(new.new_node_no1, pos=new.new_node_pos)
        graph.add_node(new.new_node_no2, pos=new.new_node_pos)
        graph.add_edge(new.upstream_node_no, new.new_node_no1)
        graph.add_edge(new.new_node_no2, new.downstream_node_no)

    # split_node: delete node and delete x edges, create x nodes, create x edges
    split_nodes_nodes = split_nodes[split_nodes.node_no!=-1]
    for split_node_id in split_nodes_nodes.node_no.values:
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
    split_nodes: gpd.GeoDataFrame,
):
    """add basin (subgraph) code to nodes and edges"""
    print(" - define numbers Ribasim-Basins and join edges/nodes")
    subgraphs = list(nx.weakly_connected_components(graph))
    if nodes is None or edges is None:
        return None, None
    nodes["basin"] = -1
    edges["basin"] = -1
    for i, subgraph in enumerate(subgraphs):
        node_ids = list(subgraph) + list(split_nodes.node_no.values)
        edges.loc[
            edges["from_node"].isin(node_ids) & 
            edges["to_node"].isin(node_ids),
            "basin",
        ] = i+1
        nodes.loc[nodes["node_no"].isin(list(subgraph)), "basin"] = i+1
    return nodes, edges


def check_if_split_node_is_used(split_nodes, nodes, edges):
    """check whether split_nodes are used, split_nodes and split_edges"""
    print(" - check whether each split location results in a split")
    split_nodes['status'] = True

    # check if edges connected to split_nodes have the same basin code
    split_node_ids = [v for v in split_nodes.node_no.values if v != -1]
    split_nodes_not_used = []
    for split_node_id in split_node_ids:
        from_nodes = list(edges[edges.from_node == split_node_id].to_node.values)
        to_nodes = list(edges[edges.to_node == split_node_id].from_node.values)
        neighbours = nodes[nodes.node_no.isin(from_nodes + to_nodes)]
        if len(neighbours.basin.unique()) == 1:
            split_nodes_not_used.append(split_node_id)
    split_nodes.loc[split_nodes[split_nodes.node_no.isin(split_nodes_not_used)].index, 'status'] = False

    # check if nodes connected to split_edge have the same basin code
    split_edge_ids = [v for v in split_nodes.edge_no.values if v != -1]
    split_edges_not_used = []
    for split_edge_id in sorted(split_edge_ids):
        end_nodes = list(edges[edges.edge_no == split_edge_id].to_node.values)
        start_nodes = list(edges[edges.edge_no == split_edge_id].from_node.values)
        neighbours = nodes[nodes.node_no.isin(end_nodes + start_nodes)]
        if len(neighbours.basin.unique()) == 1:
            split_edges_not_used.append(split_edge_id)
    split_nodes.loc[split_nodes[split_nodes.edge_no.isin(split_edges_not_used)].index, 'status'] = False

    split_nodes['object_type'] = split_nodes['object_type'].fillna('manual')
    split_nodes['split_type'] = split_nodes['object_type']
    split_nodes.loc[~split_nodes.status, 'split_type'] = 'no_split'
    return split_nodes


def create_basin_areas_based_on_drainage_areas(
    edges: gpd.GeoDataFrame, areas: gpd.GeoDataFrame
):
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
    basin_areas['basin'] = basin_areas['basin'].astype(int)
    basin_areas['area_ha'] = basin_areas.geometry.area / 10000.0
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
    tmp = nodes.merge(centralities, how='outer', left_on='node_no', right_on='node_id')
    tmp = tmp[tmp['basin']!=-1].sort_values(by=['basin', 'centrality'], ascending=[True, False])
    basins = tmp.groupby(by='basin').first().reset_index().set_crs(nodes.crs)
    return basins


def check_if_nodes_edges_within_basin_areas(nodes, edges, basin_areas):
    """"check whether nodes assigned to a basin are also within the polygon assigned to that basin"""
    if basin_areas is None:
        nodes['basin_area'] = -1
        nodes['basin_check'] = True
        edges['basin_area'] = -1
        edges['basin_check'] = True
        return nodes, edges
    
    nodes = nodes.drop(columns=['basin_area'], errors='ignore')
    nodes = gpd.sjoin(nodes, basin_areas[['geometry', 'basin']], how='left').drop(columns=['index_right'])
    nodes['basin_right'] = nodes['basin_right'].fillna(-1).astype(int)
    nodes = nodes.rename(columns={'basin_left': 'basin', 'basin_right': 'basin_area'})
    nodes['basin_check'] = nodes['basin']==nodes['basin_area']

    edges = edges.drop(columns=['basin_area'], errors='ignore')
    edges = gpd.sjoin(edges, basin_areas[['geometry', 'basin']], how='left', predicate='within').drop(columns=['index_right'])
    edges = edges.rename(columns={'basin_left': 'basin', 'basin_right': 'basin_area'})
    edges['basin_area'] = edges['basin_area'].fillna(-1).astype(int)
    edges['basin_check'] = edges['basin']==edges['basin_area']
    return nodes, edges


def create_basin_connections(
        split_nodes: gpd.GeoDataFrame, 
        edges: gpd.GeoDataFrame, 
        nodes: gpd.GeoDataFrame, 
        basins: gpd.GeoDataFrame,
        crs: int = 28992,
    ) -> gpd.GeoDataFrame:
    """create basin connections"""
    print(" - create Ribasim-Edges between Basins and split locations")
    conn = split_nodes[
        ['node_no','geometry', 'split_type', 'name', 'edge_no']
    ].rename(columns={"geometry":"geom_split_node"})
    # check if split_node is used (split_type)
    conn = conn[conn['split_type']!='no_split']

    # use different approach for: (1) splitnodes that are structures and on an edge and (2) splitnodes that are original d-hydro nodes

    # (1) splitnodes that are structures and on an edge
    conn_struct = conn.loc[conn['node_no']==-1]
    # merge with edge to find us and ds nodes
    conn_struct = conn_struct.merge(
        edges[['from_node', 'to_node','edge_no']],
        left_on='edge_no', 
        right_on='edge_no'
    )
    # TODO: check for each edge the maximum absolute flow direction, in case of negative, reverse from_node/to_node
    # merge with node to find us and ds basin
    conn_struct_us = conn_struct.merge(
        nodes,
        left_on='from_node', 
        right_on='node_no',
        suffixes=('','_r'),
    ).drop(columns=['from_node', 'to_node', 'node_no_r', 'geometry'])
    conn_struct_ds = conn_struct.merge(
        nodes,
        left_on='to_node', 
        right_on='node_no',
        suffixes=('','_r')
    ).drop(columns=['from_node', 'to_node', 'node_no_r', 'geometry'])
    
    # (2) splitnodes that are original d-hydro nodes
    # merge splitnodes add connected edges
    conn_nodes = conn.loc[conn['node_no']!=-1].drop(columns=['edge_no'])

    conn_nodes_ds = conn_nodes.merge(
        edges[['basin', 'from_node', 'to_node','edge_no']],
        left_on='node_no', 
        right_on='from_node'
    )
    conn_nodes_us = conn_nodes.merge(
        edges[['basin', 'from_node','to_node','edge_no']],
        left_on='node_no', 
        right_on='to_node'
    )

    # TODO: check for each edge the maximum absolute flow direction, in case of negative, cut and past in other dataframe.

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

    conn_ds_edge = conn_ds[conn_ds.split_type == 'edge'].copy()
    conn_us_edge = conn_us[conn_us.split_type == 'edge'].copy()

    # merge up- and downstream
    conn_edge_gdf = conn_us_edge.merge(
        conn_ds_edge, 
        left_on='edge_no',
        right_on='edge_no',
        suffixes=('_out','_in')
    )
    conn_edge_gdf = conn_edge_gdf[[
        'split_type_out', 'edge_no', 'basin_out', 'basin_in',
        'geom_basin_out', 'geom_split_node_out', 'geom_basin_in'
    ]]
    # draw connection line via split node
    if conn_edge_gdf.empty:
        conn_edge_gdf['geometry'] = np.nan
    else:
        conn_edge_gdf['geometry'] = conn_edge_gdf.apply(
            lambda row: LineString([row['geom_basin_out'], row['geom_split_node_out'], row['geom_basin_in']]), 
            axis=1
        )
    conn_edge_gdf = gpd.GeoDataFrame(conn_edge_gdf, geometry='geometry', crs=crs)
    conn_edge_gdf['name'] = -1
    
    conn_ds_node = conn_ds[conn_ds.split_type != 'edge'].copy()
    conn_us_node = conn_us[conn_us.split_type != 'edge'].copy()

    # merge up- and downstream
    conn_node_gdf = conn_us_node.merge(
        conn_ds_node, 
        left_on='name',
        right_on='name',
        suffixes=('_out','_in')
    )
    conn_node_gdf = conn_node_gdf[[
        'split_type_out', 'name', 'basin_out', 'basin_in',
        'geom_basin_out', 'geom_split_node_out', 'geom_basin_in'
    ]]
    # draw connection line via split node
    if conn_node_gdf.empty:
        conn_node_gdf['geometry'] = np.nan
    else:
        conn_node_gdf['geometry'] = conn_node_gdf.apply(
            lambda row: LineString([row['geom_basin_out'], row['geom_split_node_out'], row['geom_basin_in']]), 
            axis=1
        )
    conn_node_gdf = gpd.GeoDataFrame(conn_node_gdf, geometry='geometry', crs=crs)
    conn_node_gdf['edge_no'] = -1

    basin_connections_gdf = pd.concat([conn_edge_gdf, conn_node_gdf])
    
    basin_connections_gdf = basin_connections_gdf.drop(columns=['geom_basin_in','geom_basin_out','geom_split_node_out'])
    return basin_connections_gdf


def create_boundary_connections(
        boundaries: gpd.GeoDataFrame, 
        nodes: gpd.GeoDataFrame, 
        basins: gpd.GeoDataFrame,
    ) -> gpd.GeoDataFrame:
    """create boundary-basin connections"""
    print(" - create Ribasim-Edges between Boundary and Basin")
    if boundaries is None or nodes is None or basins is None:
        return None
    display(boundaries)
    # merge boundaries with edges 
    boundary_conn = boundaries.rename(columns={"geometry":"geometry_boundary"})
    boundary_conn_us = boundary_conn.merge(
        nodes[['node_id', 'basin']], 
        left_on='node_id', 
        right_on='node_id'
    )
    boundary_conn_ds = boundary_conn.merge(
        nodes[['node_id', 'basin']], 
        left_on='node_id', 
        right_on='node_id'
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

    boundary_conn_gdf = boundary_conn_gdf.drop(columns=['geometry_boundary','geometry_basin'])
    boundary_conn_gdf.insert(0, 'boundary_conn_id', range(len(boundary_conn_gdf)))
    boundary_conn_gdf = boundary_conn_gdf[
        ['boundary_conn_id', 'network_node_id', 'basin', 'boundary_location', 'geometry']
    ]
    return boundary_conn_gdf


def create_basins_and_connections_using_split_nodes(
    nodes: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
    split_nodes: gpd.GeoDataFrame,
    areas: gpd.GeoDataFrame,
    boundaries: gpd.GeoDataFrame,
    crs: int = 28992
) -> Dict:
    """create basins (nodes) and basin_areas (large polygons) and connections (edges) 
    based on nodes, edges, split_nodes and areas (discharge units). 
    This function calls all other functions"""
    print("Create basins using split nodes:")
    network_graph = None
    basin_areas = None
    basins = None
    network_graph = create_graph_based_on_nodes_edges(nodes, edges)
    network_graph = split_graph_based_on_split_nodes(network_graph, split_nodes, edges)
    nodes, edges = add_basin_code_from_network_to_nodes_and_edges(network_graph, nodes, edges, split_nodes)
    split_nodes = check_if_split_node_is_used(split_nodes, nodes, edges)
    basins = create_basins_based_on_subgraphs_and_nodes(network_graph, nodes)
    areas, basin_areas = create_basin_areas_based_on_drainage_areas(edges, areas)
    nodes, edges = check_if_nodes_edges_within_basin_areas(nodes, edges, basin_areas)
    boundary_connections = create_boundary_connections(boundaries, nodes, basins)
    basin_connections = create_basin_connections(split_nodes, edges, nodes, basins, crs)

    return dict(
        basin_areas=basin_areas, 
        basins=basins, 
        areas=areas, 
        nodes=nodes, 
        edges=edges, 
        split_nodes=split_nodes, \
        network_graph=network_graph, 
        basin_connections=basin_connections,
        boundary_connections=boundary_connections
    )

