import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.ops import nearest_points


def find_nearest_nodes(
    search_locations: gpd.GeoDataFrame, 
    nodes: gpd.GeoDataFrame, 
    id_column: str
) -> gpd.GeoDataFrame:
    nearest_node_ids = []
    for index, row in search_locations.iterrows():
        point = row.geometry
        multipoint = nodes.drop(index, axis=0).geometry.unary_union
        _, nearest_geom = nearest_points(point, multipoint)
        nearest_node = nodes.loc[nodes["geometry"] == nearest_geom]
        nearest_node_ids.append(nearest_node[id_column].iloc[0])
    nearest_points = gpd.GeoDataFrame(
        data={id_column: nearest_node_ids},
        geometry=search_locations['geometry'],
        crs=search_locations.crs
    )
    return nearest_points


def find_nearest_edges(
    search_locations: gpd.GeoDataFrame, 
    edges: gpd.GeoDataFrame, 
    id_column: str, 
    tolerance: int = 100,
    crs: int = 28992
) -> gpd.GeoDataFrame:
    """Function to find nearest linestring including nearest location on edge"""
    bbox = search_locations.bounds + [-tolerance, -tolerance, tolerance, tolerance]
    hits = bbox.apply(lambda row: list(edges.sindex.intersection(row)), axis=1)    
    tmp = pd.DataFrame({
        "pt_idx": np.repeat(hits.index, hits.apply(len)),
        "line_i": np.concatenate(hits.values)
    })
    tmp = tmp.join(edges.reset_index(drop=True), on="line_i")
    tmp = tmp.join(search_locations.geometry.rename("point"), on="pt_idx")
    tmp = gpd.GeoDataFrame(tmp, geometry="geometry", crs=search_locations.crs)
    
    tmp["snap_dist"] = tmp.geometry.distance(gpd.GeoSeries(tmp.point))
    tmp = tmp.loc[tmp.snap_dist <= tolerance]
    tmp = tmp.sort_values(by=["snap_dist"])

    nearest_points = tmp.groupby("pt_idx").first()
    nearest_points = (nearest_points[[id_column, 'point']]
                      .rename(columns={'point': 'geometry'})
                      .reset_index(drop=True))
    nearest_points = gpd.GeoDataFrame(nearest_points, geometry="geometry", crs=crs)
    return nearest_points

