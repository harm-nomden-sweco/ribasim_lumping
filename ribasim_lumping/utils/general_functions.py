import os
from pathlib import Path
import configparser
from typing import Dict, List, Tuple, Union
from collections import OrderedDict
import numpy as np
import pandas as pd
import geopandas as gpd
from pydantic import BaseModel
from shapely.ops import nearest_points, snap
from shapely.geometry import LineString, Point


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


def get_point_parallel_to_line_near_point(
    line: LineString, 
    reference_point: Point, 
    side: str = 'left', 
    distance: int = 5
):
    parallel_line = line.parallel_offset(distance, 'left')
    new_point = snap(reference_point, parallel_line, distance*1.1)
    return new_point


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
    ind_gdf1, ind_gdf2  = gdf2['geometry'].sindex.nearest(gdf1['geometry'], return_all=False)
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


def read_geom_file(
        filepath: Path, 
        layer_name: str = None,
        crs: int = 28992,
        explode_geoms: bool = True,
        remove_z_dim: bool = False
    ) -> gpd.GeoDataFrame:
    """
    Read file with geometries. If geopackage, supply layer_name.

    Parameters
    ----------
    filepath : Path
        Path to file containing geometries
    layer_name : str
        Layer name in geopackage. Needed when file is a geopackage
    crs : int
        CRS EPSG code. Default 28992 (RD New)
    explode_geoms : bool
        Explode multi-part geometries into single part. Default True
    remove_z_dim : bool
        Remove Z dimension from geometries. Only possible for single part Point and LineString.
        Default False

    Returns
    -------
    GeoDataFrame
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Could not find file {os.path.abspath(filepath)}")
    if str(filepath).lower().endswith('.gpkg'):
        gdf = gpd.read_file(filepath, layer=layer_name, crs=crs)
    else:
        gdf = gpd.read_file(filepath, crs=crs)
    if explode_geoms:
        gdf = gdf.explode()  # explode to transform multi-part geoms to single
    if remove_z_dim:
        gdf.geometry = [Point(g.coords[0][:2]) if isinstance(g, Point) else LineString([c[:2] for c in g.coords])
                        for g in gdf.geometry.values]  # remove possible Z dimension
    return gdf


def generate_nodes_from_edges(
        edges: gpd.GeoDataFrame
    ) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Generate start/end nodes from edges and update node information in edges GeoDataFrame.
    Return updated edges geodataframe and nodes geodataframe

    Parameters
    ----------
    edges : gpd.GeoDataFrame
        Line feature dataset containing edges

    Returns
    -------
    Tuple containing GeoDataFrame with edges and GeoDataFrame with nodes
    """

    print('Generate nodes from edges...')
    edges['edge_no'] = range(len(edges))
    edges.index = edges['edge_no']

    # Generate nodes from edges and include extra information in edges
    edges[["from_node", "to_node"]] = [[g.coords[0], g.coords[-1]] for g in edges.geometry]  # generate endpoints
    _nodes = pd.unique(edges["from_node"].tolist() + edges["to_node"].tolist())  # get unique nodes
    indexer = dict(zip(_nodes, range(len(_nodes))))
    nodes = gpd.GeoDataFrame(
        data={'node_no': [indexer[x] for x in _nodes]}, 
        index=[indexer[x] for x in _nodes], 
        geometry=[Point(x) for x in _nodes],
        crs=edges.crs
    )
    edges[["from_node", "to_node"]] = edges[["from_node", "to_node"]].map(indexer.get)  # get node id instead of coords
    return edges, nodes


def snap_to_network(
        snap_type: str,
        points: gpd.GeoDataFrame, 
        edges: gpd.GeoDataFrame = None,
        nodes: gpd.GeoDataFrame = None,
        buffer_distance: float = 0.5
    ) -> gpd.GeoDataFrame:
    """
    Snap point geometries to network based on type and within buffer distance

    Parameters
    ----------
    snap_type : str
        Snap type which control how geometries will be snapped to network. Can either be "split_node" or "boundary".
    points : gpd.GeoDataFrame
        Point feature dataset containing points to be snapped
    edges : gpd.GeoDataFrame
        Line feature dataset containing edges of network
    nodes : gpd.GeoDataFrame
        Point feature dataset containing nodes of network
    buffer_distance: float
        Buffer distance (in meter) that is used to snap nodes to network

    Returns
    -------
    GeoDataFrame with snapped geometries that are either snapped or not (based on edge_no or node_no column value)
    """

    if snap_type == "split_node":
        print(f"Snapping split nodes to nodes or edges within buffer distance ({buffer_distance:.3f} m)...")
        points = snap_points_to_nodes_and_edges(
            points, 
            edges=edges, 
            nodes=nodes, 
            edges_bufdist=buffer_distance,
            nodes_bufdist=buffer_distance * 0.1,
            n_edges_to_node_limit=3,
        )
        # print out all non-snapped split nodes
        if any([(n == -1) and (e == -1) for n, e in zip(points['node_no'], points['edge_no'])]):
            print(f"The following split nodes could not be snapped to nodes or edges within buffer distance ({buffer_distance:.3f} m):")
            for i, row in points.iterrows():
                if (row['node_no'] == -1) and (row['edge_no'] == -1):
                    print(f"  Split node {row['split_node']} - split_node_id {row['split_node_id']}")
        return points
    elif snap_type == "boundary":
        print(f"Snapping boundaries to nodes within buffer distance ({buffer_distance:.3f} m)...")
        points = snap_points_to_nodes_and_edges(
            points, 
            edges=None,   # exclude edges on purpose
            nodes=nodes, 
            nodes_bufdist=buffer_distance,
        )
        # print out all non-snapped boundaries
        if any([(n == -1) for n in points['node_no']]):
            print(f"The following boundaries could not be snapped to nodes within buffer distance ({buffer_distance:.3f} m):")
            for i, row in points.iterrows():
                if row['node_no'] == -1:
                    print(f"  Boundary {row['boundary_id']} - {row['boundary_naam']}")
        return points
    else:
        raise ValueError('Invalid snap_type. Can either be "split_node" or "boundary"')


def snap_points_to_nodes_and_edges(
        points: gpd.GeoDataFrame, 
        edges: gpd.GeoDataFrame = None,
        nodes: gpd.GeoDataFrame = None,
        edges_bufdist: float = 0.5,
        nodes_bufdist: float = 0.5,
        n_edges_to_node_limit: int = 1e10,
    ) -> gpd.GeoDataFrame:
    """
    Snap point geometries to network based on type and within buffer distance

    Parameters
    ----------
    points : gpd.GeoDataFrame
        Point feature dataset containing points to be snapped
    edges : gpd.GeoDataFrame
        Line feature dataset containing edges. Use None to don't snap point to edges. Note: if nodes are supplied
        too, this snapping will first try to snap to nodes and if not possible, to edges.
    nodes : gpd.GeoDataFrame
        Point feature dataset containing nodes. Use None to don't snap point to nodes
    edges_bufdist : float
        Buffer distance (in meter) that is used to snap point to edges
    nodes_bufdist : float
        Buffer distance (in meter) that is used to snap points to nodes
    n_edges_to_node_limit : int
        Limit the snapping to node by the number of edges that is connected to that node. There is no 
        snapping to node if number of connected edges to node is greater or equal than this value 
        (no snapping if n_connected_edges >= n_edges_to_node_limit). In order to use this option
        both nodes and edges need to be supplied.

    Returns
    -------
    GeoDataFrame with snapped points (whether or not it's snapped can be derived from edge_no or node_no column value)
    """

    print(f"Snapping points to nodes and/or edges")
    new_points = points.geometry.tolist()
    for i, point in enumerate(points.geometry):
        if nodes is not None:
            check = False
            # check if point is within buffer distance of node(s)
            ix = nodes.index.values[nodes.intersects(point.buffer(nodes_bufdist))]
            if len(nodes.loc[ix]) >= 1:
                _dist_n_to_nodes = np.array([n.distance(point) 
                                             for n in nodes.loc[ix].geometry.values])
                _ix = ix[np.argmin(_dist_n_to_nodes)]
                new_point = nodes.loc[_ix, 'geometry']
                node_no = nodes.loc[_ix, 'node_no']
                if edges is not None:
                    # first try with know edge/node info for speed-up, otherwise find the connected edges
                    if 'node_no' in edges.columns:
                        _edges = edges.loc[node_no == 'node_no']
                    else:
                        _edges = edges.loc[new_point.buffer(0.000001).intersects(edges.geometry.values)]
                    # if connected edges less than limit, snap to node
                    if len(_edges) < n_edges_to_node_limit:
                        check = True
                    else:
                        print(f"  DEBUG - Point with index {points.index.values[i]} can be snapped to node no {node_no} "
                              f"but number of connected edges to node ({len(_edges)}) is equal or higher than limit "
                              f"({n_edges_to_node_limit}). Don't snap to node and try to snap to edge. Please inspect manually")
                else:
                    check = True
            # if check is True, a valid node for point to snap to has been found
            if check:
                new_points[i] = new_point
                continue  # no need to check snapping to edge in this case
        if edges is not None:
            # if no edge is within point combined with buffer distance, skip
            lines = edges.geometry.values[edges.geometry.intersects(point.buffer(edges_bufdist))]
            # also skip if line is shorter than 2 m
            lines = lines[[l.length > 2 for l in lines]]
            if len(lines) == 0:
                continue
            # project node onto edge but make sure resulting point is some distance (0.5 meter) from start/end node of edge
            _dist_along_line = [l.project(point) for l in lines]
            _dist_along_line = [((l.length - 0.5) if (d > (l.length - 0.5)) else d) if (d > 0.5) else 0.5 
                                for l, d in zip(lines, _dist_along_line)]
            _nodes = np.array([l.interpolate(d) for l, d in zip(lines, _dist_along_line)], dtype=object)
            _dist_n_to_nodes = np.array([n.distance(point) for n in _nodes])
            # filter out nodes that is within buffer distance and use one with minimum distance
            ix = np.where(_dist_n_to_nodes <= edges_bufdist)[0]
            if len(ix) >= 1:
                # select snapped node that is closest to edge
                _nodes, _dist_n_to_nodes = _nodes[ix], _dist_n_to_nodes[ix]
                new_points[i] = _nodes[np.argmin(_dist_n_to_nodes)]
            else:
                pass   # no snapping to edge could be achieved

    # overwrite geoms with newly snapped point locations
    points['geometry'] = new_points
    # get node no or edge no on which point is located
    points = get_node_no_and_edge_no_for_points(points, edges=edges, nodes=nodes)

    return points


def get_node_no_and_edge_no_for_points(
        points: gpd.GeoDataFrame, 
        edges: gpd.GeoDataFrame = None,
        nodes: gpd.GeoDataFrame = None
    ) -> gpd.GeoDataFrame:
    """
    Get edge no or node no for point locations. If value is -1 no node and/or edge could be found for point location. 

    Parameters
    ----------
    points : gpd.GeoDataFrame
        Point feature dataset containing point locations
    nodes : gpd.GeoDataFrame
        Point feature dataset containing nodes. Note that it tries to find node first and if not possible tries to find edge
    edges : gpd.GeoDataFrame
        Line feature dataset containing edges

    Returns
    -------
    Original GeoDataFrame of split nodes with extra edge_no and node_no column
    """

    print('Retrieving edge no or node no for point locations...')
    prev_typs = None
    for typ, gdf in zip(['node_no', 'edge_no'], [nodes, edges]):
        if gdf is not None:
            gdf_no = np.ones(len(points), dtype=int) * -1
            gdf_ix = np.arange(len(gdf))
            gdf_buf = gdf.geometry.values.buffer(0.000001)  # for speed-up
            for i, point in enumerate(points.geometry):
                # skip if previous checked types resulted in valid result
                check = False
                if prev_typs is not None:
                    for prev_typ in prev_typs:
                        if points.iloc[i][prev_typ] != -1:
                            check = True
                if check:
                    continue
                # do below if not skipped
                ix = gdf_ix[gdf_buf.intersects(point)]
                if len(ix) >= 1:
                    gdf_no[i] = gdf.iloc[ix[0]][typ]  # only use first one found
            points[typ] = gdf_no
            prev_typs = prev_typs + [typ] if isinstance(prev_typs, list) else [typ]
        else:
            points[typ] = [None] * len(points)  # fill with None
    return points


def split_edges_by_split_nodes(
        split_nodes: gpd.GeoDataFrame, 
        edges: gpd.GeoDataFrame,
        buffer_distance: float = 0.5
    ) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Splits edges (lines) by split node locations. Split nodes should be (almost) perfectly be aligned to edges (within buffer distance).
    If not, use .snap_nodes_to_edges() before to align them to edges within a buffer distance. 
    
    If split nodes gdf contains edge_no column (which is filled with only integers), only those edges will be split. If the column is missing
    from gdf or contains None values, it will be ignored and the default (more time consuming) approach will be used.

    The start/end nodes will be regenerated after the edges are split. The edge no and node no column value in split nodes gdf will also 
    be updated because of that. Returns new (splitted) edges and new updated (start/end)nodes and split nodes for those edges

    Parameters
    ----------
    split_nodes : gpd.GeoDataFrame
        Point feature dataset containing split nodes
    edges : gpd.GeoDataFrame
        Line feature dataset containing edges
    buffer_distance: float
        Buffer distance (in meter) that is used project split nodes to edge

    Returns
    -------
    Tuple containing GeoDataFrame with split nodes, GeoDataFrame with edges and GeoDataFrame (start/end)nodes of edges
    """

    print("Split edges by split nodes locations...")
    split_nodes['edge_no'] = [None] * len(split_nodes)
    edge_no_col_present = 'edge_no' in split_nodes.columns
    edge_no_col_present = all([x is not None for x in split_nodes['edge_no']]) if edge_no_col_present else False
    edges_orig = edges.copy()
    # to speed-up splitting and if edge_no column is present in split nodes gdf, only
    # split those edges
    if edge_no_col_present:
        for edge_no in split_nodes['edge_no'].unique():
            if edge_no == -1:
                continue  # skip
            split_points = split_nodes.loc[split_nodes['edge_no'] == edge_no].geometry.values
            edge = edges_orig.loc[edges_orig['edge_no'] == edge_no, 'geometry'].values[0]
            splitted_edges = split_line_in_multiple(edge, distances_along_line=[edge.project(p) for p in split_points])
            if len(splitted_edges) == 0:
                continue  # skip because edge is (somehow) not splitted
            # update (original) edges gdf
            edge_row = edges_orig.loc[edges_orig['edge_no'] == edge_no]
            edges_to_add = pd.concat([edge_row]*len(splitted_edges))
            edges_to_add = gpd.GeoDataFrame(
                edges_to_add, 
                geometry=splitted_edges
            ).set_index(np.arange(len(splitted_edges)) + 1 + edges.index.max())
            edges = pd.concat([edges, edges_to_add], axis=0, ignore_index=True)
            edges.drop(index=edges_orig.loc[edges_orig['edge_no'] == edge_no].index)
    # otherwise, do default approach
    else:
        # loop over edges so we can directly split an edge with multiple split nodes in one go
        for i, edge in enumerate(edges_orig.geometry):
            # only check split nodes that are within buffer distance of edge
            split_points = split_nodes.geometry.values[split_nodes.geometry.intersects(edge.buffer(buffer_distance))]
            if len(split_points) == 0:
                continue  # skip if no split nodes found
            # also skip split nodes that are located within buffer distance from start/end nodes of edge
            nodes = np.array([edge.interpolate(x) for x in [0, edge.length]], dtype=object)
            split_points = np.array([p for p in split_points 
                                     if not any([p.intersects(n.buffer(buffer_distance)) 
                                                 for n in nodes])], 
                                    dtype=object)
            if len(split_points) == 0:
                continue  # skip if no split nodes are left
            # split edge
            splitted_edges = split_line_in_multiple(edge, distances_along_line=[edge.project(p) for p in split_points])
            if len(splitted_edges) == 0:
                continue  # skip because edge is (somehow) not splitted
            # update (original) edges gdf
            edge_row = edges_orig.loc[edges_orig.index.values[i]].to_frame().T
            edges_to_add = pd.concat([edge_row]*len(splitted_edges))
            edges_to_add = gpd.GeoDataFrame(
                edges_to_add,
                geometry=splitted_edges
            ).set_index(np.arange(len(splitted_edges)) + 1 + edges.index.max())
            edges = pd.concat([edges, edges_to_add], axis=0)
            edges.drop(index=edges_orig.index.values[i], inplace=True)

    # update edge id column if present
    if 'edge_id' in edges.columns:
        n_max = np.max(np.unique(edges['edge_id'], return_counts=True)[1])  # max group size in groupby
        split_nrs = np.arange(start=1, stop=n_max+1)
        split_nrs = edges.groupby('edge_id')['from_node'].transform(lambda x: split_nrs[:len(x)])
        max_splits = edges.groupby('edge_id')['from_node'].transform(lambda x: len(x))
        edges['edge_id'] = [f'{b}_{s}' if m > 1 else b for b, s, m in zip(edges['edge_id'], split_nrs, max_splits)]
    # regenerate start/end nodes of edges
    edges['edge_no'] = range(len(edges))  # reset edge no
    edges, nodes = generate_nodes_from_edges(edges)
    # update edge no and node no columns in split nodes gdf
    split_nodes = get_node_no_and_edge_no_for_points(split_nodes, edges, nodes)
    return split_nodes, edges, nodes


def split_line_in_two(line: LineString, distance_along_line: float) -> List[LineString]:
    # Cuts a line in two at a distance from the line starting point
    if distance_along_line <= 0.0 or distance_along_line >= line.length:
        return [LineString(line)]
    coords = list(line.coords)
    for i, p in enumerate(coords):
        pd = line.project(Point(p))
        if pd == distance_along_line:
            return [LineString(coords[:i+1]), LineString(coords[i:])]
        if pd > distance_along_line:
            cp = line.interpolate(distance_along_line)
            if len(coords[:i][0]) == 2:
                return [LineString(coords[:i] + [(cp.x, cp.y)]), LineString([(cp.x, cp.y)] + coords[i:])]
            else:
                return [LineString(coords[:i] + [(cp.x, cp.y, cp.z)]), LineString([(cp.x, cp.y, cp.z)] + coords[i:])]


def split_line_in_multiple(line: LineString, distances_along_line: Union[List[Union[float, int]], np.ndarray]) -> List[LineString]:
    # Cuts a line in multiple sections at distances from the line starting point
    lines = []
    distances_along_line = sorted(distances_along_line)  # distances should by in sorted order for loop below to work
    for i, d in enumerate(distances_along_line):
        if i == 0:
            ls = split_line_in_two(line, distances_along_line[i])
            if len(distances_along_line) == 1:
                lines = ls
                break
            else:
                lines.append(ls[0])
                new_line = ls[1]
        else:
            new_d = distances_along_line[i] - distances_along_line[i-1]
            if new_d == 0.0:
                continue
            ls = split_line_in_two(new_line, new_d)
            if i == (len(distances_along_line) - 1):
                lines.extend(ls)
            else:
                lines.append(ls[0])
                new_line = ls[1]
    return lines
