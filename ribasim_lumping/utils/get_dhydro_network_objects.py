import geopandas as gpd
import numpy as np
import pandas as pd
from typing import Dict, List


def create_objects_gdf(
    data: Dict,
    xcoor: List[float],
    ycoor: List[float],
    nodes_gdf: gpd.GeoDataFrame,
    crs: int = 28992,
):
    gdf = gpd.GeoDataFrame(
        data=data, geometry=gpd.points_from_xy(xcoor, ycoor), crs=crs
    )
    return gdf.sjoin(nodes_gdf).drop(columns=["index_right"])


def get_dhydro_network_objects(map_data, his_data, crs):
    """Extracts nodes, edges, confluences, bifurcations, weirs, pumps, laterals from his/map"""
    if map_data is None:
        raise ValueError("D-Hydro simulation map-data is not read")
    if his_data is None:
        raise ValueError("D-Hydro simulation his-data is not read")
    nodes_gdf = get_nodes_dhydro_network(map_data, crs)
    edges_gdf = get_edges_dhydro_network(map_data, crs)
    weirs_gdf = get_weirs_dhydro_network(his_data, nodes_gdf, crs)
    pumps_gdf = get_pumps_dhydro_network(his_data, nodes_gdf, crs)
    laterals_gdf = get_laterals_dhydro_network(his_data, nodes_gdf, crs)
    confluences_gdf = get_confluences_dhydro_network(nodes_gdf, edges_gdf)
    bifurcations_gdf = get_bifurcations_dhydro_network(nodes_gdf, edges_gdf)
    print(
        "dhydro-network analysed: nodes/edges/confluences/bifurcations/weirs/pumps/laterals"
    )
    return nodes_gdf, edges_gdf, weirs_gdf, pumps_gdf, laterals_gdf, \
        confluences_gdf, bifurcations_gdf


def get_nodes_dhydro_network(map_data, crs) -> gpd.GeoDataFrame:
    """calculate nodes dataframe"""
    nodes_gdf = (
        map_data["mesh1d_node_id"]
        .ugrid.to_geodataframe()
        .reset_index()
        .set_crs(crs)
    )
    nodes_gdf["mesh1d_node_id"] = nodes_gdf["mesh1d_node_id"].astype(str)
    return nodes_gdf


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
    return edges_gdf


def get_confluences_dhydro_network(nodes_gdf, edges_gdf) -> gpd.GeoDataFrame:
    """calculate confluence points based on finding multiple inflows"""
    c = edges_gdf.end_node_no.value_counts()
    confluences_gdf = nodes_gdf[
        nodes_gdf.index.isin(c.index[c.gt(1)])
    ]
    return confluences_gdf


def get_bifurcations_dhydro_network(nodes_gdf, edges_gdf) -> gpd.GeoDataFrame:
    """calculate split points based on finding multiple outflows"""
    d = edges_gdf.start_node_no.value_counts()
    bifurcations_gdf = nodes_gdf[
        nodes_gdf.index.isin(d.index[d.gt(1)])
    ]
    return bifurcations_gdf


def get_weirs_dhydro_network(his_data, nodes_gdf, crs) -> gpd.GeoDataFrame:
    """Get weirs from dhydro_model"""
    weirs_gdf = create_objects_gdf(
        data={"weirgen": his_data["weirgens"]},
        xcoor=his_data["weirgen_geom_node_coordx"].data[::2],
        ycoor=his_data["weirgen_geom_node_coordy"].data[::2],
        nodes_gdf=nodes_gdf[["mesh1d_nNodes", "geometry"]],
        crs=crs,
    )
    return weirs_gdf


def get_pumps_dhydro_network(his_data, nodes_gdf, crs) -> gpd.GeoDataFrame:
    """Get pumps from dhydro_model"""
    pumps_gdf = create_objects_gdf(
        data={"pumps": his_data["pumps"]},
        xcoor=his_data["pump_geom_node_coordx"].data[::2],
        ycoor=his_data["pump_geom_node_coordy"].data[::2],
        nodes_gdf=nodes_gdf[["mesh1d_nNodes", "geometry"]],
        crs=crs,
    )
    return pumps_gdf


def get_laterals_dhydro_network(his_data, nodes_gdf, crs) -> gpd.GeoDataFrame:
    """Get laterals from dhydro_model"""
    laterals_gdf = create_objects_gdf(
        data={"lateral": his_data["lateral"]},
        xcoor=his_data["lateral_geom_node_coordx"],
        ycoor=his_data["lateral_geom_node_coordy"],
        nodes_gdf=nodes_gdf[["mesh1d_nNodes", "geometry"]],
        crs=crs,
    )
    return laterals_gdf
