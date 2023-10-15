"""
Read network locations from D-Hydro simulation
Harm Nomden (Sweco)
"""
from pathlib import Path
import configparser
import pandas as pd
import geopandas as gpd
import hydrolib.core.dflowfm as hcdfm
from shapely.geometry import Point, LineString
import xarray as xr
from .general_functions import find_file_in_directory, \
    find_directory_in_directory, get_points_on_linestrings_based_on_distances, \
        replace_string_in_file, read_ini_file_with_similar_sections, find_nearest_edges_no


def get_dhydro_files(simulation_path: Path):
    """Get DHydro input files"""
    input_files = dict()
    mdu_file = ""
    mdu_file = find_file_in_directory(simulation_path, ".mdu")
    print(f"  - MDU-file: {mdu_file}")
    replace_string_in_file(mdu_file, "*\n", "# *\n")
    mdu = configparser.ConfigParser()
    mdu_dir = Path(mdu_file).parent
    mdu.read(mdu_file)
    replace_string_in_file(mdu_file, "# *\n", "*\n")

    input_files["mdu_file"] = mdu_file
    input_files["net_file"] = Path(mdu_dir, mdu["geometry"]["netfile"])
    input_files["structure_file"] = Path(mdu_dir, mdu["geometry"]["structurefile"])
    input_files["cross_loc_file"] = Path(mdu_dir, mdu["geometry"]["crosslocfile"])
    input_files["cross_def_file"] = Path(mdu_dir, mdu["geometry"]["crossdeffile"])
    input_files["friction_file"] = Path(mdu_dir, mdu["geometry"]["frictfile"])
    input_files["external_forcing_file"] = Path(
        mdu_dir, mdu["external forcing"]["extforcefilenew"]
    )
    input_files["obs_file"] = Path(mdu_dir, mdu["output"]["obsfile"])

    output_dir = mdu["output"]["outputdir"]
    input_files["output_dir"] = Path(find_directory_in_directory(simulation_path, output_dir))
    input_files["output_his_file"] = Path(find_file_in_directory(simulation_path, "his.nc"))
    input_files["output_map_file"] = Path(find_file_in_directory(simulation_path, "map.nc"))
    return input_files


def get_dhydro_network_data(network_file: Path):
    """Get DHydro network locations"""
    print("  - network:", end="", flush=True)
    return hcdfm.net.models.Network.from_file(network_file), xr.open_dataset(network_file)


def get_dhydro_branches_from_network_data(network_data, crs):
    """Get DHydro branches"""
    print(" branches", end="", flush=True)
    branch_keys = [b for b in network_data._mesh1d.branches.keys()]
    branch_geom = [b.geometry for b in network_data._mesh1d.branches.values()]
    branches_df = pd.DataFrame({"branchid": branch_keys, "branchgeom": branch_geom})
    branches_df["geometry"] = branches_df.apply(
        lambda row: LineString(row["branchgeom"]), axis=1
    )
    branches_gdf = gpd.GeoDataFrame(
        branches_df, 
        geometry="geometry", 
        crs=crs
    ).drop("branchgeom", axis=1)
    return branches_gdf


def get_dhydro_network_nodes_from_network_nc(network_nc, crs):
    try:
        nodes_df = pd.DataFrame({
            "network_node_id": network_nc.network_node_id,
            "X": network_nc.network_node_x,
            "Y": network_nc.network_node_y,
        })
    except:
        nodes_df = pd.DataFrame({
            "network_node_id": network_nc.Network_node_id,
            "X": network_nc.Network_node_x,
            "Y": network_nc.Network_node_y,
        })
    nodes_df["geometry"] = list(zip(nodes_df["X"], nodes_df["Y"]))
    nodes_df["geometry"] = nodes_df["geometry"].apply(Point)
    nodes_gdf = gpd.GeoDataFrame(nodes_df, geometry="geometry", crs=crs)
    nodes_gdf['network_node_id'] = nodes_gdf['network_node_id'].apply(lambda x: str(x)[2:-14])
    return nodes_gdf.drop(['X', 'Y'], axis=1)


def get_dhydro_nodes_from_network_data(network_data, crs):
    """Get DHydro nodes"""
    print(" nodes", end="", flush=True)
    nodes_df = pd.DataFrame({
        "branch_id": network_data._mesh1d.mesh1d_node_branch_id,
        "node_id": network_data._mesh1d.mesh1d_node_id,
        "geometry": list(zip(network_data._mesh1d.mesh1d_node_x, network_data._mesh1d.mesh1d_node_y))
    })
    nodes_df["node_no"] = nodes_df.index
    nodes_df["geometry"] = nodes_df["geometry"].apply(Point)
    nodes_gdf = gpd.GeoDataFrame(nodes_df, geometry="geometry", crs=crs)
    return nodes_gdf


def get_dhydro_edges_from_network_data(network_data, nodes_gdf, crs):
    """Get DHydro edges"""
    print(" edges")#, end="", flush=True)
    edges_df = pd.DataFrame({
        "branch_id": network_data._mesh1d.mesh1d_edge_branch_id,
        "X": network_data._mesh1d.mesh1d_edge_x,
        "Y": network_data._mesh1d.mesh1d_edge_y,
        "from_node": network_data._mesh1d.mesh1d_edge_nodes[:, 0],
        "to_node": network_data._mesh1d.mesh1d_edge_nodes[:, 1],
    })
    edges_df["geometry"] = ""
    edges_gdf = edges_df.merge(
        nodes_gdf,
        how="inner",
        left_on="from_node",
        right_on="node_no",
        suffixes=["", "_from"],
    )
    edges_gdf = edges_gdf.merge(
        nodes_gdf,
        how="inner",
        left_on="to_node",
        right_on="node_no",
        suffixes=["", "_to"],
    )

    edges_gdf["geometry"] = edges_gdf.apply(
        lambda row: LineString([row["geometry_from"], row["geometry_to"]]), axis=1
    )
    edges_gdf = gpd.GeoDataFrame(edges_gdf, geometry="geometry", crs=crs)
    edges_gdf["edge_no"] = edges_gdf.index
    edges_gdf = edges_gdf[
        ["edge_no", "branch_id", "geometry", "from_node", "to_node"]
    ]
    return edges_gdf


def get_dhydro_structures_locations(
    structures_file: Path, 
    branches_gdf: gpd.GeoDataFrame,
    edges_gdf: gpd.GeoDataFrame
):
    """Get DHydro structures locations"""
    # get structure file (e.g. "structures.ini")
    print("  - structures:", end="", flush=True)
    m = hcdfm.structure.models.StructureModel(structures_file)
    structures_df = pd.DataFrame([f.__dict__ for f in m.structure]).set_index("id")
    structures_gdf = get_points_on_linestrings_based_on_distances(
        linestrings=branches_gdf,
        linestring_id_column='branchid',
        points=structures_df,
        points_linestring_id_column='branchid',
        points_distance_column='chainage'
    )
    structures_gdf = structures_gdf.rename(columns={"type": "object_type"})
    structures_gdf = find_nearest_edges_no(
        gdf1=structures_gdf, 
        gdf2=edges_gdf.set_index('edge_no').sort_index(),
        new_column='edge_no',
    )
    return structures_gdf


def split_dhydro_structures(structures_gdf: gpd.GeoDataFrame):
    """Get all DHydro structures dataframes"""
    list_structure_types = list(structures_gdf['object_type'].unique())
    structures_gdf_dict = {}
    for structure_type in list_structure_types:
        if structure_type == 'compound':
            continue
        print(f" {structure_type}", end="", flush=True)
        structure_gdf = structures_gdf.loc[structures_gdf["object_type"] == structure_type]
        if 'comments' in structure_gdf.columns:
            structure_gdf.loc[:, 'comments'] = structure_gdf.loc[:, 'comments'].astype(str)
        structures_gdf_dict[structure_type] = structure_gdf.dropna(axis=1, how='all')
    print(f" ")
    return structures_gdf_dict


def get_dhydro_external_forcing_locations(
    external_forcing_file: str, 
    branches_gdf: gpd.GeoDataFrame, 
    network_nodes_gdf: gpd.GeoDataFrame,
    nodes_gdf: gpd.GeoDataFrame
):
    """Get all DHydro boundaries and laterals"""
    print("  - external forcing (locations):", end="", flush=True)
    
    print(" boundaries", end="", flush=True)
    boundaries_gdf = read_ini_file_with_similar_sections(external_forcing_file, 'Boundary')
    boundaries_gdf = network_nodes_gdf[['network_node_id', 'geometry']].merge(
        boundaries_gdf.rename(columns={'nodeid': 'network_node_id'}), 
        how='inner', 
        left_on='network_node_id', 
        right_on='network_node_id'
    ).sjoin(nodes_gdf[["node_id", "node_no", "geometry"]]).drop('index_right', axis=1)

    print(" laterals")
    laterals_gdf = read_ini_file_with_similar_sections(external_forcing_file, 'Lateral')
    laterals_gdf['chainage'] = laterals_gdf['chainage'].astype(float)
    laterals_gdf = get_points_on_linestrings_based_on_distances(
        linestrings=branches_gdf,
        linestring_id_column='branchid',
        points=laterals_gdf,
        points_linestring_id_column='branchid',
        points_distance_column='chainage'
    )
    return boundaries_gdf, laterals_gdf


def get_dhydro_forcing_data(
    boundaries_gdf: gpd.GeoDataFrame, 
    laterals_gdf: gpd.GeoDataFrame
):
    """Get DHydro forcing data"""
    print("  - external forcing (data):", end="", flush=True)
    
    print(" boundaries", end="", flush=True)
    # boundary_data = None
    # if boundary_file is None:
    #     print(" * simulation has no boundary file")
    # else:
    #     boundary_filepath = Path(simulation_path, boundary_file)
    #     forcingmodel_object = hcdfm.ForcingModel(boundary_filepath)
    #     boundary_data = pd.DataFrame(
    #         [forcing.dict() for forcing in forcingmodel_object.forcing]
    #     )
    #     # convert dictionary with boundary type to columns
    #     boundary_data = pd.concat(
    #         [
    #             boundary_data.drop(["quantityunitpair"], axis=1),
    #             pd.DataFrame.from_records(boundary_data["quantityunitpair"])[0].apply(
    #                 pd.Series
    #             ),
    #         ],
    #         axis=1,
    #     )
    boundaries_data = None

    print(" laterals")
    # get laterals timeseries (e.g. "FlowFM_lateral_sources.bc")
    # lateral_filepath = Path(simulation_path, lateral_file)
    # forcingmodel_object = hcdfm.ForcingModel(lateral_filepath)
    # lateral_data = pd.DataFrame(
    #     [forcing.dict() for forcing in forcingmodel_object.forcing]
    # )
    laterals_data = None
    return boundaries_data, laterals_data


def get_dhydro_data_from_simulation(dhydro_dir: Path, simulation_name: str, crs):
    """Get DHydro data from simulation"""
    simulation_path = Path(dhydro_dir, simulation_name)

    network_data = None
    network_nodes_gdf = None
    files = None
    branches_gdf = None
    nodes_gdf = None
    edges_gdf = None
    structures_gdf = None
    structures_dict = None
    boundaries_gdf = None
    laterals_gdf = None
    laterals_data = None
    boundaries_data = None

    files = get_dhydro_files(simulation_path)

    network_data, network_nc = get_dhydro_network_data(files['net_file'])
    network_nodes_gdf = get_dhydro_network_nodes_from_network_nc(network_nc, crs)
    branches_gdf = get_dhydro_branches_from_network_data(network_data, crs)
    nodes_gdf = get_dhydro_nodes_from_network_data(network_data, crs)
    edges_gdf = get_dhydro_edges_from_network_data(network_data, nodes_gdf, crs)

    structures_gdf = get_dhydro_structures_locations(files['structure_file'], branches_gdf, edges_gdf)
    structures_dict = split_dhydro_structures(structures_gdf)

    boundaries_gdf, laterals_gdf = get_dhydro_external_forcing_locations(
        files['external_forcing_file'], 
        branches_gdf, 
        network_nodes_gdf, 
        nodes_gdf
    )
    boundaries_data, laterals_data = get_dhydro_forcing_data(boundaries_gdf, laterals_gdf)
    
    results = dict(
        network_data=network_data,
        network_nodes_gdf=network_nodes_gdf,
        files=files,
        branches_gdf=branches_gdf,
        nodes_gdf=nodes_gdf,
        edges_gdf=edges_gdf,
        structures_gdf=structures_gdf,
        structures_dict=structures_dict,
        boundaries_gdf=boundaries_gdf,
        laterals_gdf=laterals_gdf,
        laterals_data=laterals_data,
        boundaries_data=boundaries_data
    )

    return results
