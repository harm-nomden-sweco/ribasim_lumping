# pylint: disable=missing-function-docstring
import os
from pathlib import Path
from typing import List, Union, Optional, Any, Tuple, Dict
import shutil
import datetime
from pydantic import BaseModel, Field
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import polylabel
import pandas as pd
import numpy as np
import dfm_tools as dfmt
import xarray as xr
import xugrid as xu
import networkx as nx
from shapely.ops import nearest_points

from .utils.read_simulation_data_utils import (
    get_data_from_simulations_set,
    get_simulation_names_from_dir,
    combine_data_from_simulations_sets,
)
from .utils.generate_basins_areas import create_basins_based_on_split_node_ids


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


class RibasimLumpingNetwork(BaseModel):
    """class to select datapoints from different simulations at certain timestamps"""

    name: str
    his_data: xu.UgridDataset = None
    map_data: xu.UgridDataset = None
    edges_gdf: gpd.GeoDataFrame = None
    nodes_gdf: gpd.GeoDataFrame = None
    network_graph: nx.DiGraph = None
    areas_gdf: gpd.GeoDataFrame = None
    basin_areas_gdf: gpd.GeoDataFrame = None
    basins_gdf: gpd.GeoDataFrame = None
    confluences_gdf: gpd.GeoDataFrame = None
    bifurcations_gdf: gpd.GeoDataFrame = None
    weirs_gdf: gpd.GeoDataFrame = None
    pumps_gdf: gpd.GeoDataFrame = None
    laterals_gdf: gpd.GeoDataFrame = None
    culverts_gdf: gpd.GeoDataFrame = None
    split_node_ids: List[int] = []
    split_nodes: gpd.GeoDataFrame = None
    crs: int = 28992

    class Config:
        arbitrary_types_allowed = True

    def add_data_from_simulations_set(
        self,
        set_name: str,
        simulations_dir: Path,
        simulations_names: List[str] = None,
        simulations_ts: Union[List, pd.DatetimeIndex] = [-1],
    ) -> Tuple[xr.Dataset, xu.UgridDataset]:
        """ "receives his- and map-data
        - from d-hydro simulations with names: simulation_names
        - within directory: simulations_dir
        - at timestamps: simulations_ts"""
        if simulations_names is None:
            if not Path(simulations_dir).exists():
                raise ValueError(
                    f"Directory D-Hydro calculations does not exist: {simulations_dir}"
                )
            self.simulation_names = get_simulation_names_from_dir(simulations_dir)

        his_data, map_data = get_data_from_simulations_set(
            set_name=set_name,
            simulations_dir=simulations_dir,
            simulations_names=simulations_names,
            simulations_ts=simulations_ts,
        )
        self.his_data = combine_data_from_simulations_sets(self.his_data, his_data)
        self.map_data = combine_data_from_simulations_sets(
            self.map_data, map_data, xugrid=True
        )
        return self.his_data, self.map_data

    def get_network_data(self):
        """Gives fast access to nodes, edges, confluences, bifurcations, weirs, pumps, laterals"""
        if self.map_data is None:
            raise ValueError("D-Hydro simulation map-data is not read")
        if self.his_data is None:
            raise ValueError("D-Hydro simulation his-data is not read")
        self.get_nodes()
        self.get_edges()
        self.get_confluences_gdf()
        self.get_bifurcations_gdf()
        self.get_weirs()
        self.get_pumps()
        self.get_laterals()
        print(
            "network locations read: nodes/edges/confluences/bifurcations/weirs/pumps/laterals"
        )

    def get_nodes(self) -> gpd.GeoDataFrame:
        """calculate nodes dataframe"""
        self.nodes_gdf = (
            self.map_data["mesh1d_node_id"]
            .ugrid.to_geodataframe()
            .reset_index()
            .set_crs(self.crs)
        )
        self.nodes_gdf["mesh1d_node_id"] = self.nodes_gdf["mesh1d_node_id"].astype(str)
        return self.nodes_gdf

    def get_edges(self) -> gpd.GeoDataFrame:
        """calculate edges dataframe"""
        edges = (
            self.map_data["mesh1d_q1"][-1][-1]
            .ugrid.to_geodataframe()
            .reset_index()
            .set_crs(self.crs)
            .drop(columns=["condition", "mesh1d_q1", "set"])
        )
        edges_nodes = self.map_data["mesh1d_edge_nodes"]
        edges_nodes = np.column_stack(edges_nodes.data)
        edges_nodes = pd.DataFrame(
            {"start_node_no": edges_nodes[0] - 1, "end_node_no": edges_nodes[1] - 1}
        )
        self.edges_gdf = edges.merge(
            edges_nodes, how="inner", left_index=True, right_index=True
        )
        return self.edges_gdf

    def get_node_ids_from_type(
        self,
        bifurcations: bool = False,
        confluences: bool = False,
        weirs: bool = False,
        laterals: bool = False,
    ):
        split_node_ids = []
        if bifurcations and self.bifurcations_gdf is not None:
            split_node_ids += list(self.bifurcations_gdf.mesh1d_nNodes.values)
        if confluences and self.confluences_gdf is not None:
            split_node_ids += list(self.confluences_gdf.mesh1d_nNodes.values)
        if weirs and self.weirs_gdf is not None:
            split_node_ids += list(self.weirs_gdf.mesh1d_nNodes.values)
        if laterals and self.laterals_gdf is not None:
            split_node_ids += list(self.laterals_gdf.mesh1d_nNodes.values)
        return split_node_ids

    def add_split_nodes_based_on_node_ids(
        self, 
        split_node_ids: List[int], 
        level: int = 0
    ) -> gpd.GeoDataFrame:
        if level < 1 or level > 10:
            raise ValueError('add split_node_ids at a level between 1 and 10')
        split_nodes = self.nodes_gdf[
            self.nodes_gdf.mesh1d_nNodes.isin(split_node_ids)
        ].reset_index(drop=True)
        split_nodes[f'level_{level}'] = 1

        if self.split_nodes is None:
            self.split_nodes = split_nodes
        else:
            if f'level_{level}' in self.split_nodes.columns:
                self.split_nodes = self.split_nodes.drop(columns=f'level_{level}')
            split_nodes = pd.concat([self.split_nodes, split_nodes])
            col_groupby = 'mesh1d_nNodes'
            cols_merge = ['mesh1d_nNodes', 'mesh1d_node_id', 'mesh1d_node_x', 'mesh1d_node_y', 'geometry']
            cols_sum = [c for c in split_nodes.columns if c not in cols_merge]
            split_nodes = (
                split_nodes.groupby(col_groupby)
                .agg({c: 'sum' for c in cols_sum})
                .reset_index()
                .merge(split_nodes[cols_merge], how='outer', left_on=col_groupby, right_on=col_groupby)
                .drop_duplicates()
                .astype({c: 'int32' for c in cols_sum})
            )
            self.split_nodes = split_nodes[cols_merge + cols_sum]
        return self.split_nodes

    def create_basins_based_on_split_node_gdf(
        self,
        split_node_gdf: gpd.GeoDataFrame,
        areas: gpd.GeoDataFrame = None,
        nodes: gpd.GeoDataFrame = None,
        edges: gpd.GeoDataFrame = None,
    ) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        split_nodes_gdf = self.find_nearest_nodes(
            split_nodes_gdf=split_node_gdf, nodes=nodes
        )
        if split_nodes_gdf is not None:
            split_node_ids = list(split_nodes_gdf["nearest_node_id"].values)
        else:
            split_node_ids = []
        return self.create_basins_based_on_split_node_ids(
            split_node_ids=split_node_ids,
            areas=areas,
            nodes=nodes,
            edges=edges,
        )

    def create_basins_based_on_split_node_ids(
        self,
        split_node_ids: List[int],
        areas: gpd.GeoDataFrame = None,
        nodes: gpd.GeoDataFrame = None,
        edges: gpd.GeoDataFrame = None,
    ) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        if nodes is None:
            nodes = self.nodes_gdf
        if edges is None:
            edges = self.edges_gdf
        if areas is None:
            areas = self.areas_gdf
        self.split_node_ids = split_node_ids
        basin_areas, areas, nodes, edges = create_basins_based_on_split_node_ids(
            nodes=self.nodes_gdf,
            edges=self.edges_gdf,
            split_node_ids=split_node_ids,
            areas=areas,
        )
        self.nodes_gdf = nodes
        self.edges_gdf = edges
        self.areas_gdf = areas
        self.basin_areas_gdf = basin_areas

        self.get_basins()

        return self.basin_areas_gdf, self.areas_gdf, self.nodes_gdf, self.edges_gdf

    def get_confluences_gdf(self) -> gpd.GeoDataFrame:
        """calculate confluence points based on finding multiple inflows"""
        c = self.edges_gdf.end_node_no.value_counts()
        self.confluences_gdf = self.nodes_gdf[
            self.nodes_gdf.index.isin(c.index[c.gt(1)])
        ]
        return self.confluences_gdf

    def get_bifurcations_gdf(self) -> gpd.GeoDataFrame:
        """calculate split points based on finding multiple outflows"""
        d = self.edges_gdf.start_node_no.value_counts()
        self.bifurcations_gdf = self.nodes_gdf[
            self.nodes_gdf.index.isin(d.index[d.gt(1)])
        ]
        return self.bifurcations_gdf

    def get_weirs(self) -> gpd.GeoDataFrame:
        self.weirs_gdf = create_objects_gdf(
            data={"weirgen": self.his_data["weirgens"]},
            xcoor=self.his_data["weirgen_geom_node_coordx"].data[::2],
            ycoor=self.his_data["weirgen_geom_node_coordy"].data[::2],
            nodes_gdf=self.nodes_gdf[["mesh1d_nNodes", "geometry"]],
            crs=self.crs,
        )
        return self.weirs_gdf

    def get_pumps(self) -> gpd.GeoDataFrame:
        self.pumps_gdf = create_objects_gdf(
            data={"pumps": self.his_data["pumps"]},
            xcoor=self.his_data["pump_geom_node_coordx"].data[::2],
            ycoor=self.his_data["pump_geom_node_coordy"].data[::2],
            nodes_gdf=self.nodes_gdf[["mesh1d_nNodes", "geometry"]],
            crs=self.crs,
        )
        return self.pumps_gdf

    def get_laterals(self) -> gpd.GeoDataFrame:
        self.laterals_gdf = create_objects_gdf(
            data={"lateral": self.his_data["lateral"]},
            xcoor=self.his_data["lateral_geom_node_coordx"],
            ycoor=self.his_data["lateral_geom_node_coordy"],
            nodes_gdf=self.nodes_gdf[["mesh1d_nNodes", "geometry"]],
            crs=self.crs,
        )
        return self.laterals_gdf

    def get_basins(self):
        if self.basin_areas_gdf is None:
            self.basins_gdf = None
        else:
            self.basin_areas_gdf["area"] = self.basin_areas_gdf.geometry.area
            self.basins_gdf = self.basin_areas_gdf.copy()
            self.basins_gdf["geometry"] = [
                polylabel(polygon) for polygon in self.basins_gdf.geometry.values
            ]
            self.basins_gdf = self.basins_gdf.set_crs(self.crs)
        return self.basins_gdf

    def export_to_geopackage(
        self,
        output_dir: Union[Path, str],
    ):
        dir_output = Path(output_dir, self.name)
        if not dir_output.exists():
            dir_output.mkdir()
        
        gpkg_path = Path(dir_output, "ribasim_network.gpkg")
        qgz_path = Path(dir_output, "ribasim_network.qgz")

        qgz_path_stored_dir = os.path.abspath(os.path.dirname(__file__))
        qgz_path_stored = Path(qgz_path_stored_dir, "assets\\ribasim_network.qgz")
        
        if self.split_node_ids is not None and self.nodes_gdf is not None:
            split_nodes = self.nodes_gdf.loc[self.split_node_ids]
            split_nodes["split_node_id"] = split_nodes["mesh1d_nNodes"]
            split_nodes = split_nodes[["split_node_id", "geometry"]]
        else:
            split_nodes = None

        gdfs = dict(
            nodes=self.nodes_gdf,
            edges=self.edges_gdf,
            confluences=self.confluences_gdf,
            bifurcations=self.bifurcations_gdf,
            weirs=self.weirs_gdf,
            pumps=self.pumps_gdf,
            laterals=self.laterals_gdf,
            areas=self.areas_gdf,
            basin_areas=self.basin_areas_gdf,
            split_nodes=split_nodes,
            basins=self.basins_gdf,
        )
        for gdf_name, gdf in gdfs.items():
            if gdf is None:
                gpd.GeoDataFrame(
                    columns=['geometry'],
                    geometry='geometry', 
                    crs=self.crs
                ).to_file(gpkg_path, layer=gdf_name, driver="GPKG")
            else:
                gdf.to_file(gpkg_path, layer=gdf_name, driver="GPKG")
        if not qgz_path.exists():
            shutil.copy(qgz_path_stored, qgz_path)

    def find_nearest_nodes(
        self, split_nodes_gdf: gpd.GeoDataFrame, nodes: gpd.GeoDataFrame = None
    ) -> gpd.GeoDataFrame:
        # split_nodes must have columns: 'splitnode_id','geometry'
        if nodes is None:
            nodes = self.nodes_gdf
        if nodes is None:
            return None

        for index, row in split_nodes_gdf.iterrows():
            point = row.geometry
            multipoint = nodes.drop(index, axis=0).geometry.unary_union
            queried_geom, nearest_geom = nearest_points(point, multipoint)
            # split_nodes.loc[index, 'nearest_geometry'] = nearest_geom

            nearest_node = nodes.loc[nodes["geometry"] == nearest_geom]
            split_nodes_gdf.loc[index, "nearest_node_id"] = nearest_node[
                "mesh1d_nNodes"
            ].iloc[0]
        split_nodes_gdf.nearest_node_id = split_nodes_gdf.nearest_node_id.astype(int)
        split_nodes_gdf = split_nodes_gdf.set_crs(28992)
        return split_nodes_gdf

    # def get_qh_relations_weirs(self, weirs_ids: List[str] = None):
    #     # TODO: get QH relations from selected locations
    #     return weirs_ids

    # def get_qh_relation_edges_around_node(
    #     self, nodes_ids: List[str] = None, nodes_locations: List[Point] = None
    # ):
    #     # TODO: get QH relations from selected locations
    #     return nodes_ids
