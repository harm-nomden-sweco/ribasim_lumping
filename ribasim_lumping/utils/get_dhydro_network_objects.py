import geopandas as gpd
import numpy as np
import pandas as pd
from typing import Dict, List
from .general_functions import find_nearest_nodes, find_nearest_edges


def create_objects_gdf(
    data: Dict,
    xcoor: List[float],
    ycoor: List[float],
    edges_gdf: gpd.GeoDataFrame,
    crs: int = 28992,
):
    gdf = gpd.GeoDataFrame(
        data=data, 
        geometry=gpd.points_from_xy(xcoor, ycoor), 
        crs=crs
    )
    nearest_points = find_nearest_edges(
        search_locations=gdf, 
        edges=edges_gdf, 
        id_column='mesh1d_nEdges',
        crs=crs
    )
    gdf = gpd.GeoDataFrame(
        data=(gdf.drop(columns='geometry')
              .merge(nearest_points, how='outer', left_index=True, right_index=True)),
        geometry='geometry',
        crs=crs
    )
    return gdf


def get_dhydro_network_objects(map_data, his_data, crs):
    """Extracts nodes, edges, confluences, bifurcations, weirs, pumps, laterals from his/map"""
    if map_data is None:
        raise ValueError("D-Hydro simulation map-data is not read")
    if his_data is None:
        raise ValueError("D-Hydro simulation his-data is not read")
    print("D-HYDRO-network analysed:")
    nodes, nodes_h = get_nodes_dhydro_network(map_data, crs)
    print(" - nodes and waterlevels")
    edges, edges_q = get_edges_dhydro_network(map_data, crs)
    print(" - edges and discharges")
    weirs = get_weirs_dhydro_network(his_data, edges, crs)
    print(" - weirs")
    uniweirs = get_uniweirs_dhydro_network(his_data, edges, crs)
    print(" - uniweirs")
    pumps = get_pumps_dhydro_network(his_data, edges, crs)
    print(" - pumps")
    confluences = get_confluences_dhydro_network(nodes, edges)
    print(" - confluences")
    bifurcations = get_bifurcations_dhydro_network(nodes, edges)
    print(" - bifurcations")
    return nodes, nodes_h, edges, edges_q, weirs, uniweirs, pumps, confluences, bifurcations


def get_nodes_dhydro_network(map_data, crs) -> gpd.GeoDataFrame:
    """calculate nodes dataframe"""
    nodes_gdf = (
        map_data["mesh1d_node_id"]
        .ugrid.to_geodataframe()
        .reset_index()
        .set_crs(crs)
    ).drop(columns=['mesh1d_node_x', 'mesh1d_node_y'])
    nodes_gdf["mesh1d_node_id"] = nodes_gdf["mesh1d_node_id"].astype(str).apply(lambda r: r[2:-1].strip())
    nodes_h_df = map_data.mesh1d_s1.to_dataframe()[['mesh1d_s1']]
    nodes_h_df = nodes_h_df.reorder_levels(['mesh1d_nNodes', 'set', 'condition'])
    return nodes_gdf, nodes_h_df


def get_edges_dhydro_network(map_data, crs) -> gpd.GeoDataFrame:
    """calculate edges dataframe"""
    edges = (
        map_data["mesh1d_q1"][-1][-1]
        .ugrid.to_geodataframe()
        .reset_index()
        .set_crs(crs)
        .drop(columns=["condition", "mesh1d_q1", "set"])
    )
    edges_nodes = map_data["mesh1d_edge_nodes"]
    edges_nodes = np.column_stack(edges_nodes.data)
    edges_nodes = pd.DataFrame(
        {"start_node_no": edges_nodes[0] - 1, "end_node_no": edges_nodes[1] - 1}
    )
    edges_gdf = edges.merge(
        edges_nodes, how="inner", left_index=True, right_index=True
    )
    edges_q_df = map_data.mesh1d_q1.to_dataframe()[['mesh1d_q1']]
    edges_q_df = edges_q_df.reorder_levels(['mesh1d_nEdges', 'set', 'condition'])
    return edges_gdf, edges_q_df


def get_confluences_dhydro_network(nodes_gdf, edges_gdf) -> gpd.GeoDataFrame:
    """calculate confluence points based on finding multiple inflows"""
    c = edges_gdf.end_node_no.value_counts()
    confluences_gdf = nodes_gdf[
        nodes_gdf.index.isin(c.index[c.gt(1)])
    ].reset_index(drop=True)
    confluences_gdf.object_type = 'confluence'
    return confluences_gdf


def get_bifurcations_dhydro_network(nodes_gdf, edges_gdf) -> gpd.GeoDataFrame:
    """calculate split points based on finding multiple outflows"""
    d = edges_gdf.start_node_no.value_counts()
    bifurcations_gdf = nodes_gdf[
        nodes_gdf.index.isin(d.index[d.gt(1)])
    ].reset_index(drop=True)
    bifurcations_gdf.object_type = 'bifurcation'
    return bifurcations_gdf


def get_weirs_dhydro_network(his_data, edges_gdf, crs) -> gpd.GeoDataFrame:
    """Get weirs from dhydro_model"""
    weirs_gdf = create_objects_gdf(
        data={"mesh1d_node_id": his_data.weirgens},
        xcoor=his_data.weir_input_geom_node_coordx,
        ycoor=his_data.weir_input_geom_node_coordy,
        edges_gdf=edges_gdf[["mesh1d_nEdges", "geometry"]],
        crs=crs,
    )
    weirs_gdf['object_type'] = 'weir'
    return weirs_gdf


def get_uniweirs_dhydro_network(his_data, edges_gdf, crs) -> gpd.GeoDataFrame:
    """Get weirs from dhydro_model"""
    uniweirs_gdf = create_objects_gdf(
        data={"mesh1d_node_id": his_data.universalWeirs},
        xcoor=his_data.uniweir_input_geom_node_coordx,
        ycoor=his_data.uniweir_input_geom_node_coordy,
        edges_gdf=edges_gdf[["mesh1d_nEdges", "geometry"]],
        crs=crs,
    )
    uniweirs_gdf['object_type'] = 'uniweir'
    return uniweirs_gdf


def get_pumps_dhydro_network(his_data, edges_gdf, crs) -> gpd.GeoDataFrame:
    """Get pumps from dhydro_model"""
    pumps_gdf = create_objects_gdf(
        data={"mesh1d_node_id": his_data.pumps},
        xcoor=his_data.pump_input_geom_node_coordx,
        ycoor=his_data.pump_input_geom_node_coordy,
        edges_gdf=edges_gdf[["mesh1d_nEdges", "geometry"]],
        crs=crs,
    )
    pumps_gdf['object_type'] = 'pump'
    return pumps_gdf

