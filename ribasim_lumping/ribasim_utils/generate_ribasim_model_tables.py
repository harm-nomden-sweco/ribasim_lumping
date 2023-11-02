import pandas as pd
import geopandas as gpd
import numpy as np


def generate_basin_static_table(basin_h, basin_a, basins, decimals=3):
    basins_node_id = basins[['basin', 'node_id']].set_index('basin')['node_id']
    basin_profile = pd.DataFrame(
        columns=['node_id', 'level', 'area', 'remarks']
    )
    for basin_no in basin_h.columns:
        basin_profile_col = basin_h[[basin_no]].reset_index(drop=True).round(decimals)
        basin_profile_col.columns = ['level']
        basin_profile_col['area'] = basin_a[[basin_no]].reset_index(drop=True).round(decimals)
        basin_profile_col['node_id'] = basins_node_id.loc[basin_no]
        basin_profile = pd.concat([basin_profile, basin_profile_col])
    basin_profile["remarks"] = ""
    basin_profile = basin_profile[basin_profile['level'].notna()]
    return basin_profile.reset_index(drop=True)


def generate_basin_time_table(basins, basin_areas, start_date="2020-01-01", end_date="2020-1-31"):
    time = pd.date_range(start_date, end_date)
    day_of_year = time.day_of_year.to_numpy()
    seconds_per_day = 24 * 60 * 60

    evaporation = 0.0
    precipitation = 0.0
    # evaporation = (
    #     (-1.0 * np.cos(day_of_year / 365.0 * 2 * np.pi) + 1.0) * 0.0025 / seconds_per_day
    # )
    # rng = np.random.default_rng(seed=0)
    # precipitation = (
    #     rng.lognormal(mean=-1.0, sigma=1.7, size=time.size) * 0.001 / seconds_per_day
    # )
    basin_areas_ha = basin_areas.set_index('node_id')

    timeseries = pd.DataFrame()
    for basin_no in basins.node_id.values:
        if basin_no in basin_areas_ha.index:
            area = basin_areas_ha.loc[basin_no, 'area_ha']
            drainage = 4.0 / 1000 * area
        else:
            drainage = 0.0
        timeseries = pd.concat([
            timeseries,
            pd.DataFrame(
                data={
                    "node_id": basin_no,
                    "time": time,
                    "drainage": drainage,
                    "potential_evaporation": evaporation,
                    "infiltration": 0.0,
                    "precipitation": precipitation * 10.0,
                    "urban_runoff": 0.0,
                }
            ).sort_values(['time', 'node_id'])
        ])
    return timeseries


def generate_tabulate_rating_curve(basins_outflows, split_nodes):
    basins_outflows = basins_outflows.sort_values(by='split_node')
    basins_outflows['split_node_node_id'] = split_nodes['node_id']
    basins_outflows = basins_outflows[['split_node_node_id', 'crestlevel', 'crestwidth']].sort_values(by='split_node_node_id')
    curves = pd.DataFrame()
    for i, weir in basins_outflows.dropna(how='any', axis=0).iterrows():
        levels = [weir["crestlevel"] + i for i in [-1.0, 0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.4, 0.5, 0.75, 1.00, 2.00, 4.00]]

        def weir_formula(crestlevel, crestwidth, waterlevel):
            return 2.0/3.0 * max(0.0, waterlevel - crestlevel)**(3.0/2.0) * (2*9.81)**0.5 * crestwidth

        discharges = [weir_formula(weir["crestlevel"], weir["crestwidth"], level) for level in levels]
        curve = pd.DataFrame(
            data={
                "node_id": weir["split_node_node_id"],
                "level": levels,
                "discharge": discharges
            }
        )
        curves = pd.concat([curves, curve])
    curves['node_id'] = curves['node_id'].astype(int)
    return curves


def generate_ribasim_model_tables(basin_h, basin_a, basins, basin_areas, boundaries, \
    boundaries_data, split_nodes, basins_outflows, set_name):

    # create tables for BASINS
    tables = dict()
    tables['basin_profile'] = generate_basin_static_table(basin_h, basin_a, basins, decimals=3)
    tables['basin_time'] = generate_basin_time_table(basins, basin_areas)
    tables['basin_state'] = pd.DataFrame(
        data={
            "node_id": basins.node_id.values,
            "level": basin_h.loc[(set_name, 'targetlevel')].values,
        }
    )

    # create tables for BOUNDARIES
    level_boundaries = boundaries[boundaries['ribasim_type']=="LevelBoundary"]
    tables['level_boundary_static'] = pd.DataFrame(
        data={
            "node_id": level_boundaries.node_id,
            "level": [7.15] * len(level_boundaries),
        }
    )
    flow_boundaries = boundaries[boundaries['ribasim_type']=="FlowBoundary"]
    tables['flow_boundary_static'] = pd.DataFrame(
        data={
            "node_id": flow_boundaries.node_id,
            "flow_rate": [0.0] * len(flow_boundaries),
        }
    )
    
    # create tables for PUMPS
    pumps = split_nodes[split_nodes['ribasim_type'] == 'Pump']
    tables['pump_static'] = pd.DataFrame(
        data={
            "node_id": pumps.node_id,
            "flow_rate": [0] * len(pumps),
        }
    )
    # create tables for OUTLETS
    outlets = split_nodes[split_nodes['ribasim_type'] == 'Outlet']
    tables['outlet_static'] = pd.DataFrame(
        data={
            "node_id": outlets.node_id,
            "flow_rate": [0] * len(outlets),
        }
    )
    # create tables for TABULATED RATING CURVES
    tabulated_rating_curves = split_nodes[split_nodes['ribasim_type'] == 'TabulatedRatingCurve']
    tables['tabulated_rating_curve_static'] = generate_tabulate_rating_curve(basins_outflows, split_nodes)
    
    # create tables for MANNINGRESISTANCE
    manningresistance = split_nodes[split_nodes['ribasim_type'] == 'ManningResistance']
    tables['manningresistance_static'] = pd.DataFrame(
        data={
            "node_id": manningresistance.node_id,
            "length": [250.0]*len(manningresistance),
            "manning_n": [0.04]*len(manningresistance),
            "profile_width": [5.0]*len(manningresistance),
            "profile_slope": [3.0]*len(manningresistance),
        }
    )

    return tables
