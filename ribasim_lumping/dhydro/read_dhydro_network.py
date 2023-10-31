"""
Read network locations from D-Hydro simulation
Harm Nomden (Sweco)
"""
from pathlib import Path
import subprocess
import configparser
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
import xarray as xr
import xugrid as xu
import hydrolib.core.dflowfm as hcdfm
from ..utils.general_functions import find_file_in_directory, \
    find_directory_in_directory, find_nearest_nodes, get_points_on_linestrings_based_on_distances, \
        replace_string_in_file, read_ini_file_with_similar_sections, find_nearest_edges_no


def get_dhydro_files(simulation_path: Path):
    """Get DHydro input files"""
    input_files = dict()
    mdu_file = ""
    mdu_file = find_file_in_directory(simulation_path, ".mdu")
    # print(f"  - MDU-file: {mdu_file}")

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

    volume_nc_file = Path(mdu_dir, "PerGridpoint_volume.nc")
    if volume_nc_file.exists():
        input_files['volume_file'] = volume_nc_file
    else:
        input_files['volume_file'] = ""

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
    branch_keys = [b for b in network_data._mesh1d.branches.keys()]
    branch_geom = [b.geometry for b in network_data._mesh1d.branches.values()]
    branches_df = pd.DataFrame({
        "branch_id": branch_keys, 
        "branch_geom": branch_geom
    })
    branches_df["geometry"] = branches_df.apply(
        lambda row: LineString(row["branch_geom"]), axis=1
    )
    branches_gdf = gpd.GeoDataFrame(
        branches_df, 
        geometry="geometry", 
        crs=crs
    ).drop("branch_geom", axis=1)
    print(f" branches ({len(branches_gdf)}x)", end="", flush=True)
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
    nodes_gdf["network_node_id"] = nodes_gdf["network_node_id"].astype(str).str.strip(" ")
    print(f" network-nodes ({len(nodes_gdf)}x)", end="", flush=True)
    return nodes_gdf.drop(['X', 'Y'], axis=1)


def get_dhydro_nodes_from_network_data(network_data, crs):
    """Get DHydro nodes"""
    nodes_df = pd.DataFrame({
        "branch_id": network_data._mesh1d.mesh1d_node_branch_id,
        "chainage": network_data._mesh1d.mesh1d_node_branch_offset,
        "node_id": network_data._mesh1d.mesh1d_node_id,
        "geometry": list(zip(network_data._mesh1d.mesh1d_node_x, network_data._mesh1d.mesh1d_node_y))
    })
    nodes_df["node_no"] = nodes_df.index
    nodes_df["geometry"] = nodes_df["geometry"].apply(Point)
    nodes_gdf = gpd.GeoDataFrame(nodes_df, geometry="geometry", crs=crs)
    print(f" nodes ({len(nodes_gdf)}x)", end="", flush=True)
    return nodes_gdf


def get_dhydro_edges_from_network_data(network_data, nodes_gdf, crs):
    """Get DHydro edges"""
    edges_df = pd.DataFrame({
        "branch_id": network_data._mesh1d.mesh1d_edge_branch_id,
        "chainage": network_data._mesh1d.mesh1d_edge_branch_offset,
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
    print(f" edges ({len(edges_gdf)})")#, end="", flush=True)
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
    structures_df = pd.DataFrame([f.__dict__ for f in m.structure])
    structures_df = structures_df.drop('name', axis=1)
    structures_df = structures_df.rename({"branchid": "branch_id", "id": "structure_id"}, axis=1)
    structures_gdf = get_points_on_linestrings_based_on_distances(
        linestrings=branches_gdf,
        linestring_id_column='branch_id',
        points=structures_df,
        points_linestring_id_column='branch_id',
        points_distance_column='chainage'
    )
    structures_gdf = structures_gdf.rename(columns={"type": "object_type"})
    structures_gdf = find_nearest_edges_no(
        gdf1=structures_gdf, 
        gdf2=edges_gdf.set_index('edge_no').sort_index(),
        new_column='edge_no',
    )
    return structures_gdf


def check_number_of_pumps_at_pumping_station(pumps_gdf: gpd.GeoDataFrame):
    """Check number of pumps at pumping station and combine them into one representative pump
    Input:  Geodataframe with pumps with multiple per location
    Output: Geodataframe with one pump per location. 
            Total capacity (sum), Max start level, Min stop level"""
    pumps_gdf = pumps_gdf.groupby(pumps_gdf.geometry.to_wkt(), as_index=False).agg(dict(
        structure_id='first',
        branch_id='first', 
        geometry='first',
        comments='first', 
        object_type='first',
        chainage='first',
        orientation='first',
        controlside='first',
        numstages='first',
        capacity='sum',
        startlevelsuctionside='max',
        stoplevelsuctionside='min',
        startleveldeliveryside='min',
        stopleveldeliveryside='max',
        numreductionlevels='first',
        head='first',
        reductionfactor='first',
        edge_no='first'
    )).reset_index(drop=True).pipe(gpd.GeoDataFrame)
    return pumps_gdf


def split_dhydro_structures(structures_gdf: gpd.GeoDataFrame):
    """Get all DHydro structures dataframes"""
    list_structure_types = list(structures_gdf['object_type'].unique())
    structures_gdf_dict = {}
    for structure_type in list_structure_types:
        # skip all compounds
        if structure_type == "compound":
            continue
        # get structure type data
        structure_gdf = structures_gdf.loc[structures_gdf["object_type"] == structure_type]
        # comments are sometimes a separate object instead of string
        if 'comments' in structure_gdf.columns:
            structure_gdf.loc[:, 'comments'] = structure_gdf.loc[:, 'comments'].astype(str)
        # in case of pumps check if multiple pumps in one pumping station
        if structure_type == "pump" and ~structure_gdf.empty:
            old_no_pumps = len(structure_gdf)
            structure_gdf = check_number_of_pumps_at_pumping_station(structure_gdf)
            if old_no_pumps > len(structure_gdf):
                print(f" pumps ({old_no_pumps}x->{len(structure_gdf)}x)", end="", flush=True)
            else:
                print(f" pumps ({len(structure_gdf)}x)", end="", flush=True)
        else:
            print(f" {structure_type}s ({len(structure_gdf)}x)", end="", flush=True)

        structure_gdf = structure_gdf.dropna(axis=1, how='all')
        for col in structure_gdf.columns:
            if any(isinstance(val, list) for val in structure_gdf[col]):
                structure_gdf[col] = structure_gdf[col].apply(
                    lambda x: f"[{','.join([str(i) for i in x])}]" 
                    if isinstance(x, list) else ""
                )
        structures_gdf_dict[structure_type] = structure_gdf
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
    
    boundaries_gdf = read_ini_file_with_similar_sections(external_forcing_file, "Boundary")
    boundaries_gdf = boundaries_gdf.rename({"nodeid": "network_node_id"}, axis=1)
    boundaries_gdf["network_node_id"] = boundaries_gdf["network_node_id"].astype(str)
    network_nodes_gdf["network_node_id"] = network_nodes_gdf["network_node_id"].astype(str)

    boundaries_gdf = network_nodes_gdf[["network_node_id", "geometry"]].merge(
        boundaries_gdf, 
        how="right", 
        left_on="network_node_id", 
        right_on="network_node_id"
    )
    boundaries_gdf = boundaries_gdf.sjoin(find_nearest_nodes(boundaries_gdf, nodes_gdf, "node_no"))
    
    boundaries_gdf = boundaries_gdf.reset_index(drop=True)
    boundaries_gdf.insert(0, "boundary", boundaries_gdf.index + 1)
    boundaries_gdf = boundaries_gdf.rename(columns={"network_node_id": "name"})
    print(f" boundaries ({len(boundaries_gdf)}x)", end="", flush=True)

    laterals_gdf = read_ini_file_with_similar_sections(external_forcing_file, "Lateral")
    laterals_gdf["chainage"] = laterals_gdf["chainage"].astype(float)
    laterals_gdf = laterals_gdf.rename({"branchid": "branch_id"}, axis=1)
    laterals_gdf = get_points_on_linestrings_based_on_distances(
        linestrings=branches_gdf,
        linestring_id_column="branch_id",
        points=laterals_gdf,
        points_linestring_id_column="branch_id",
        points_distance_column="chainage"
    )
    print(f" laterals ({len(laterals_gdf)}x)")
    return boundaries_gdf, laterals_gdf


def get_dhydro_forcing_data(
    mdu_input_dir: Path,
    boundaries_gdf: gpd.GeoDataFrame, 
    laterals_gdf: gpd.GeoDataFrame
):
    """Get DHydro forcing data"""
    print("  - external forcing (data):", end="", flush=True)
    print(" boundaries", end="", flush=True)
    boundaries_data = None
    boundaries_forcing_files = boundaries_gdf['forcingfile'].unique()
    for boundaries_forcing_file in boundaries_forcing_files:
        boundaries_forcing_file_path = Path(mdu_input_dir, boundaries_forcing_file)
        forcingmodel_object = hcdfm.ForcingModel(boundaries_forcing_file_path)
        if boundaries_data is None:
            boundaries_data = pd.DataFrame([forcing.dict() for forcing in forcingmodel_object.forcing])
        else:
            boundaries_data = pd.concat([boundaries_data, pd.DataFrame([forcing.dict() for forcing in forcingmodel_object.forcing])])
    
    print(" laterals")
    laterals_data = None
    laterals_forcing_files = laterals_gdf['discharge'].unique()
    for laterals_forcing_file in laterals_forcing_files:
        laterals_forcing_file_path = Path(mdu_input_dir, laterals_forcing_file)
        forcingmodel_object = hcdfm.ForcingModel(laterals_forcing_file_path)
        if laterals_data is None:
            laterals_data = pd.DataFrame([forcing.dict() for forcing in forcingmodel_object.forcing])
        else:
            laterals_data = pd.concat([laterals_data, pd.DataFrame([forcing.dict() for forcing in forcingmodel_object.forcing])])
    return boundaries_data, laterals_data

def get_dhydro_volume_based_on_basis_simulations(
    mdu_input_dir: Path,
    volume_tool_bat_file: Path, 
    volume_tool_force: bool = False,
    volume_tool_increment: float = 0.1
):
    mdu_file = find_file_in_directory(mdu_input_dir, ".mdu")
    volume_nc_file = find_file_in_directory(mdu_input_dir, "PerGridpoint_volume.nc")
    if volume_nc_file == "" or volume_tool_force:
        subprocess.Popen(
            f'"{volume_tool_bat_file}" --mdufile "{mdu_file.name}" --increment {str(volume_tool_increment)} --outputfile volume.nc --output "All"', cwd=str(mdu_file.parent)
        )
        print(f"  - volume_tool: new level-volume dataframe created: {volume_nc_file.name}")
    else:
        print("  - volume_tool: file already exists, use force=True to force recalculation volume")
    volume = xu.open_dataset(volume_nc_file)
    return volume

def get_dhydro_data_from_simulation(
    simulation_path: Path, 
    volume_tool_bat_file: Path, 
    volume_tool_force: bool = False,
    volume_tool_increment: float = 0.1,
    crs: int = 28992,
):
    """Get DHydro data from simulation"""
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

    structures_gdf = get_dhydro_structures_locations(
        structures_file=files['structure_file'], 
        branches_gdf=branches_gdf, 
        edges_gdf=edges_gdf
    )
    structures_dict = split_dhydro_structures(structures_gdf)

    boundaries_gdf, laterals_gdf = get_dhydro_external_forcing_locations(
        external_forcing_file=files['external_forcing_file'], 
        branches_gdf=branches_gdf, 
        network_nodes_gdf=network_nodes_gdf, 
        nodes_gdf=nodes_gdf
    )
    mdu_input_dir = Path(files['mdu_file']).parent
    boundaries_data, laterals_data = get_dhydro_forcing_data(
        mdu_input_dir=mdu_input_dir, 
        boundaries_gdf=boundaries_gdf, 
        laterals_gdf=laterals_gdf
    )
    
    volume_data = get_dhydro_volume_based_on_basis_simulations(
        mdu_input_dir=mdu_input_dir, 
        volume_tool_bat_file=volume_tool_bat_file, 
        volume_tool_force=volume_tool_force,
        volume_tool_increment=volume_tool_increment
    )

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
        boundaries_data=boundaries_data,
        volume_data=volume_data
    )

    return results
