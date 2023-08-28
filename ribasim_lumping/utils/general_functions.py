import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.ops import nearest_points
from typing import Dict, List


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
    projected_points = gpd.GeoDataFrame(
        data={id_column: nearest_node_ids},
        geometry=search_locations['geometry'],
        crs=search_locations.crs
    )
    return projected_points


def find_nearest_edges(
    search_locations: gpd.GeoDataFrame, 
    edges: gpd.GeoDataFrame, 
    id_column: str, 
    selection: str = None,
    tolerance: int = 100,
    crs: int = 28992
) -> gpd.GeoDataFrame:
    """Function to find nearest linestring including nearest location on edge"""
    bbox = search_locations.bounds + [-tolerance, -tolerance, tolerance, tolerance]
    hits = bbox.apply(lambda row: list(edges.sindex.intersection(row)), axis=1)
    tmp = pd.DataFrame({
        "split_node_i": np.repeat(hits.index, hits.apply(len)),
        "mesh1d_nEdges": np.concatenate(hits.values)
    })
    if tmp.empty:
        return None
    if selection is not None and selection in search_locations and selection in edges:
        tmp = tmp.merge(
            search_locations.reset_index()[selection], 
            how='outer', 
            left_on='split_node_i', 
            right_index=True
        ).rename(columns={selection: f'{selection}_x'})

    tmp = tmp.merge(edges, how='inner', left_on='mesh1d_nEdges', right_on='mesh1d_nEdges')
    tmp = tmp.join(search_locations.geometry.rename("point"), on="split_node_i")
    tmp = gpd.GeoDataFrame(tmp, geometry="geometry", crs=search_locations.crs)
    
    tmp["snap_dist"] = tmp.geometry.distance(gpd.GeoSeries(tmp.point))
    tmp = tmp.loc[tmp.snap_dist <= tolerance]
    tmp = tmp.sort_values(by=["snap_dist"])

    if selection is not None and selection in search_locations and selection in edges:
        tmp = tmp[tmp[selection] == tmp[f'{selection}_x']].copy()
        tmp = tmp.drop(columns=[f'{selection}_x'])

    tmp_points = tmp.groupby("split_node_i").first()
    tmp_points['projection'] = tmp_points.apply(
        lambda x: nearest_points(x.geometry, x.point)[0], 
        axis=1
    )
    tmp_points['projection_x'] = tmp_points['projection'].apply(lambda x: x.x)
    tmp_points['projection_y'] = tmp_points['projection'].apply(lambda x: x.y)
    tmp_points = (tmp_points[[id_column, 'projection_x', 'projection_y', 'point']]
                  .rename(columns={'point': 'geometry'})
                  .reset_index(drop=True))

    projected_points = gpd.GeoDataFrame(tmp_points, geometry="geometry", crs=crs)
    return projected_points


def create_objects_gdf(
    data: Dict,
    xcoor: List[float],
    ycoor: List[float],
    edges_gdf: gpd.GeoDataFrame,
    selection: str = None,
    crs: int = 28992,
    tolerance: int = 100,
):
    gdf = gpd.GeoDataFrame(
        data=data, 
        geometry=gpd.points_from_xy(xcoor, ycoor), 
        crs=crs
    )
    projected_points = find_nearest_edges(
        search_locations=gdf, 
        edges=edges_gdf, 
        id_column='mesh1d_nEdges',
        selection=selection,
        tolerance=tolerance,
        crs=crs
    )
    if projected_points is None:
        return None
    gdf = gpd.GeoDataFrame(
        data=(gdf.drop(columns='geometry')
              .merge(projected_points, how='outer', left_index=True, right_index=True)),
        geometry='geometry',
        crs=crs
    )
    return gdf


