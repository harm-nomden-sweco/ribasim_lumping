import os
from pathlib import Path
import configparser
from typing import Dict, List
from collections import OrderedDict
import numpy as np
import pandas as pd
import geopandas as gpd
from pydantic import BaseModel
from shapely.ops import nearest_points


def replace_string_in_file(file_path, string, new_string):
    with open(file_path, "r") as file:
        content = file.read()
    content = content.replace(string, new_string)
    with open(file_path, "w") as file:
        file.write(content)


def find_file_in_directory(directory, file_name, start_end='end') -> Path:
    """Find path of file in directory"""
    selected_file = ""
    for root, dirs, files in os.walk(directory):
        for file in files:
            if start_end == 'end':
                if file.endswith(file_name):
                    selected_file = os.path.join(root, file)
            elif start_end == 'start':
                if file.startswith(file_name):
                    selected_file = os.path.join(root, file)
    return Path(selected_file)


def find_directory_in_directory(directory, dir_name) -> Path:
    """Find path of subdirectory in directory"""
    selected_dir = ""
    for root, directories, files in os.walk(directory):
        for directory in directories:
            if directory.endswith(dir_name):
                selected_dir = os.path.join(root, directory)
    return Path(selected_dir)


class MultiOrderDict(OrderedDict):
    _unique = dict()
    def __setitem__(self, key, val):
        if isinstance(val, dict):
            if key not in self._unique:
                self._unique[key] = 0
            else:
                self._unique[key] += 1
            key += str(self._unique[key])
        OrderedDict.__setitem__(self, key, val)


def read_ini_file_with_similar_sections(file_path, section_name):
    config = configparser.ConfigParser(dict_type=MultiOrderDict, strict=False)
    config.read(file_path)
    section_keys = [k for k in config.keys() if k.startswith(section_name)]
    section_name_df = pd.DataFrame([config._sections[k]  for k in section_keys])
    return section_name_df


def get_points_on_linestrings_based_on_distances(
    linestrings: gpd.GeoDataFrame, 
    linestring_id_column: str,
    points: gpd.GeoDataFrame, 
    points_linestring_id_column: str, 
    points_distance_column: str
) -> gpd.GeoDataFrame:
    """Get point location (gdf) at certain distance along linestring (gdf)"""
    points = linestrings.merge(
        points, 
        how='inner', 
        left_on=linestring_id_column, 
        right_on=points_linestring_id_column
    )
    points['geometry'] = points['geometry'].interpolate(points[points_distance_column])
    return points


def find_nearest_nodes(
    search_locations: gpd.GeoDataFrame, nodes: gpd.GeoDataFrame, id_column: str
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
        geometry=search_locations["geometry"],
        crs=search_locations.crs,
    )
    return projected_points


def find_nearest_edges_no(
    gdf1: gpd.GeoDataFrame,
    gdf2: gpd.GeoDataFrame,
    new_column: str
) -> gpd.GeoDataFrame: 
    ind_gdf1, ind_gdf2  = gdf2['geometry'].sindex.nearest(gdf1['geometry'])
    gdf1[new_column] = ind_gdf2
    return gdf1


def find_nearest_edges(
    search_locations: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
    id_column: str,
    selection: str = None,
    tolerance: int = 100,
) -> gpd.GeoDataFrame:
    """Function to find nearest linestring including nearest location on edge"""
    bbox = search_locations.bounds + [-tolerance, -tolerance, tolerance, tolerance]
    hits = bbox.apply(lambda row: list(edges.sindex.intersection(row)), axis=1)
    tmp = pd.DataFrame({
        "split_node_i": np.repeat(hits.index, hits.apply(len)),
        "edge_no": np.concatenate(hits.values),
    })
    if tmp.empty:
        return None
    if selection is not None and selection in search_locations and selection in edges:
        tmp = tmp.merge(
            search_locations.reset_index()[selection],
            how="outer",
            left_on="split_node_i",
            right_index=True,
        ).rename(columns={selection: f"{selection}_x"})
    tmp = tmp.merge(
        edges, 
        how="inner", 
        left_on="edge_no", 
        right_on="edge_no"
    )
    tmp = tmp.join(search_locations.geometry.rename("point"), on="split_node_i")
    tmp = gpd.GeoDataFrame(tmp, geometry="geometry", crs=search_locations.crs)

    tmp["snap_dist"] = tmp.geometry.distance(gpd.GeoSeries(tmp.point))
    tmp = tmp.loc[tmp.snap_dist <= tolerance]
    tmp = tmp.sort_values(by=["snap_dist"])

    if selection is not None and selection in search_locations and selection in edges:
        tmp = tmp[tmp[selection] == tmp[f"{selection}_x"]].copy()
        tmp = tmp.drop(columns=[f"{selection}_x"])

    tmp_points = tmp.groupby("split_node_i").first()
    tmp_points["projection"] = tmp_points.apply(
        lambda x: nearest_points(x.geometry, x.point)[0], axis=1
    )
    tmp_points["projection_x"] = tmp_points["projection"].apply(lambda x: x.x)
    tmp_points["projection_y"] = tmp_points["projection"].apply(lambda x: x.y)
    tmp_points = (
        tmp_points[[id_column, "projection_x", "projection_y", "point"]]
        .rename(columns={"point": "geometry"})
        .reset_index(drop=True)
    )

    projected_points = gpd.GeoDataFrame(tmp_points, geometry="geometry", crs=search_locations.crs)
    return projected_points


def create_objects_gdf(
    data: Dict,
    xcoor: List[float],
    ycoor: List[float],
    edges_gdf: gpd.GeoDataFrame,
    selection: str = None,
    tolerance: int = 100,
):
    crs = edges_gdf.crs
    gdf = gpd.GeoDataFrame(
        data=data, geometry=gpd.points_from_xy(xcoor, ycoor), crs=crs
    )
    projected_points = find_nearest_edges(
        search_locations=gdf,
        edges=edges_gdf,
        id_column="edge_no",
        selection=selection,
        tolerance=tolerance,
    )
    if projected_points is None:
        return None
    gdf = gpd.GeoDataFrame(
        data=(
            gdf.drop(columns="geometry").merge(
                projected_points, 
                how="outer", 
                left_index=True, 
                right_index=True
            )
        ),
        geometry="geometry",
        crs=crs,
    )
    return gdf
