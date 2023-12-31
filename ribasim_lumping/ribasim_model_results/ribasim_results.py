from pathlib import Path
from typing import List, Dict, Optional
import sys
import warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pydantic import BaseModel
import pandas as pd
import geopandas as gpd

from numba.core.errors import NumbaDeprecationWarning
warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=FutureWarning)
pd.options.mode.chained_assignment = None
sys.path.append("..\\..\\..\\ribasim\\python\\ribasim")
import ribasim


class RibasimModelResults(BaseModel):    
    # model: ribasim.Model
    simulation_path: Path
    ribasim_model: ribasim.Model
    node_h: pd.DataFrame
    node_v: pd.DataFrame
    basin_h: pd.DataFrame
    basin_v: pd.DataFrame
    flow_edge: Optional[pd.DataFrame] = None
    basin: Optional[pd.DataFrame] = None
    control: Optional[pd.DataFrame] = None
    flow: Optional[pd.DataFrame] = None
    basin_flow: Optional[pd.DataFrame] = None
    basin_areas: gpd.GeoDataFrame = None

    class Config:
        arbitrary_types_allowed = True


class RibasimBasinResults(BaseModel):
    ribasim_model: ribasim.Model
    basin_no: int
    basin: pd.DataFrame
    basin_profile: pd.DataFrame
    control: Optional[pd.DataFrame] = None
    control_condition: Optional[pd.DataFrame] = None
    basin_flow: Optional[pd.DataFrame] = None
    inflow_edge: Optional[pd.DataFrame] = None
    outflow_edge: Optional[pd.DataFrame] = None
    inflow: Optional[pd.DataFrame] = None
    outflow: Optional[pd.DataFrame] = None

    class Config:
        arbitrary_types_allowed = True


def read_arrow_file(simulation_dir: Path, file_name: str, results_dir: str = "results"):
    """read arrow files from folder"""
    arrow_file_path = Path(simulation_dir, results_dir, f"{file_name}.arrow")
    return pd.read_feather(arrow_file_path)


def get_inflow_outflow_edge_data(ribasim_model: ribasim.Model):
    node_static = ribasim_model.node.static
    edge_static = ribasim_model.edge.static

    basin_nos = ribasim_model.node.static[ribasim_model.node.static["type"]=="Basin"].index.values
    flow_edge = edge_static[edge_static.edge_type=="flow"]
    flow_edge = flow_edge[
        (flow_edge.to_node_id.isin(basin_nos)) | 
        (flow_edge.from_node_id.isin(basin_nos))
    ]
    
    from_node_id = flow_edge.from_node_id.values
    flow_edge['from_node_name'] = node_static["name"].loc[from_node_id]
    to_node_id = flow_edge.to_node_id.values
    flow_edge['to_node_name'] = node_static["name"].loc[to_node_id].values

    flow_edge_source = (flow_edge.set_index("to_node_id")[["from_node_id", "from_node_name"]]
                        .rename(columns={"from_node_id": "source_node_id", "from_node_name": "source_name"}))
    flow_edge = (flow_edge
                 .merge(flow_edge_source, 
                        how='left', 
                        left_on='from_node_id', 
                        right_index=True))
    flow_edge_target = (flow_edge.set_index("from_node_id")[["to_node_id", "to_node_name"]]
                        .rename(columns={"to_node_id": "target_node_id", "to_node_name": "target_name"}))
    flow_edge = (flow_edge
                 .merge(flow_edge_target, 
                        how='left', 
                        left_on='to_node_id', 
                        right_index=True))
    flow_edge.source_node_id = flow_edge.source_node_id.fillna(-1).astype(int)
    flow_edge.target_node_id = flow_edge.target_node_id.fillna(-1).astype(int)
    flow_edge.from_node_name = flow_edge.from_node_name.fillna("")
    flow_edge.to_node_name = flow_edge.to_node_name.fillna("")
    flow_edge.source_name = flow_edge.source_name.fillna("")
    flow_edge.target_name = flow_edge.target_name.fillna("")
    return flow_edge


def read_ribasim_model_results(simulation_path: Path):
    if not simulation_path.exists():
        raise ValueError(f" x simulation_path {str(simulation_path)} does not exist")
    ribasim_model = ribasim.Model.from_toml(Path(simulation_path, "ribasim.toml"))
    basin = (read_arrow_file(simulation_path, "basin")
             .set_index(["node_id", "time"])
             .sort_index()
             .drop_duplicates(keep="first"))
    basin = basin.unstack(0).interpolate().unstack(-1).unstack(0)
    control = (read_arrow_file(simulation_path, "control")
               .set_index(["control_node_id", "time"])
               .sort_index()
               .drop_duplicates(keep="first"))
    flow = (read_arrow_file(simulation_path, "flow")
            .drop_duplicates(keep="first"))
    basin_flow = flow[flow["edge_id"].isna()]
    basin_flow = (
        basin_flow
        .drop(columns=["edge_id", "from_node_id"])
        .rename(columns={"to_node_id": "node_id"})
        .set_index(["node_id", "time"]).sort_index()
    )
    flow = flow[~flow["edge_id"].isna()]
    flow["edge_id"] = flow["edge_id"].astype(int)

    flow_edge = get_inflow_outflow_edge_data(ribasim_model=ribasim_model)
    basin_areas = gpd.read_file(Path(simulation_path, "ribasim_network.gpkg"), layer="basin_areas")
    node_h = gpd.read_file(Path(simulation_path, "ribasim_network.gpkg"), layer="node_h")
    node_v = gpd.read_file(Path(simulation_path, "ribasim_network.gpkg"), layer="node_v")
    basin_h = gpd.read_file(Path(simulation_path, "ribasim_network.gpkg"), layer="basin_h")
    basin_v = gpd.read_file(Path(simulation_path, "ribasim_network.gpkg"), layer="basin_v")
    return RibasimModelResults(
        ribasim_model=ribasim_model,
        simulation_path=simulation_path, 
        node_h=node_h,
        node_v=node_v,
        basin_h=basin_h,
        basin_v=basin_v,
        flow_edge=flow_edge,
        basin=basin, 
        control=control, 
        flow=flow, 
        basin_flow=basin_flow,
        basin_areas=basin_areas,
    )


def get_ribasim_basin_data_from_model(
    ribasim_model: ribasim.Model, 
    ribasim_results: RibasimModelResults, 
    basin_no: int
):
    basin_profile = (ribasim_model.basin.profile
                     .groupby('node_id')
                     .get_group(basin_no)
                     .drop(columns=["node_id"]))
    model_basin = ribasim_results.basin.copy()
    model_control = ribasim_results.control.copy()
    model_flow = ribasim_results.flow.copy()
    model_basin_flow = ribasim_results.basin_flow.copy()

    basin = (model_basin.reset_index()
             [model_basin.reset_index().node_id==basin_no]
             .drop(columns=["node_id"])
             .set_index("time"))
    outflow = (model_flow
               [model_flow["from_node_id"]==basin_no]
               .drop(columns=["from_node_id", "edge_id"])
               .set_index(["to_node_id", "time"])
               .sort_index())
    inflow = (model_flow
              [model_flow["to_node_id"]==basin_no]
              .drop(columns=["to_node_id", "edge_id"])
              .set_index(["from_node_id", "time"])
              .sort_index())
    basin_flow = (model_basin_flow.reset_index()
                  [model_basin_flow.reset_index().node_id==basin_no]
                  .drop(columns=["node_id"])
                  .set_index("time"))

    basin = basin[~basin.index.duplicated(keep='first')]
    inflow = inflow[~inflow.index.duplicated(keep='first')]
    outflow = outflow[~outflow.index.duplicated(keep='first')]
    basin_flow = basin_flow[~basin_flow.index.duplicated(keep='first')]
    
    if ribasim_model.discrete_control is None:
        control_condition = None
        control_node_id = None
        control = None
    else:
        control_condition = (ribasim_model.discrete_control.condition
                            [ribasim_model.discrete_control.condition.listen_feature_id==basin_no])
        control_node_id = control_condition.node_id.unique()
        control = model_control.loc[control_node_id]

    inflow_edge = (ribasim_results.flow_edge[ribasim_results.flow_edge.to_node_id==basin_no]
                   .drop(columns=["target_node_id", "target_name"])
                   .drop_duplicates(keep="first")
                   .reset_index(drop=True))
    outflow_edge = (ribasim_results.flow_edge[ribasim_results.flow_edge.from_node_id==basin_no]
                    .drop(columns=["source_node_id", "source_name"])
                    .drop_duplicates(keep="first")
                    .reset_index(drop=True))
    return RibasimBasinResults(
        ribasim_model=ribasim_model,
        basin_no=basin_no,
        basin=basin, 
        basin_profile=basin_profile,
        inflow=inflow, 
        outflow=outflow, 
        basin_flow=basin_flow, 
        control=control, 
        control_condition=control_condition,
        inflow_edge=inflow_edge,
        outflow_edge=outflow_edge,
    )


def plot_results_basins_ribasim_model(
    ribasim_model: ribasim.Model, 
    simulation_path: Path, 
):
    ribasim_results = read_ribasim_model_results(simulation_path=simulation_path)
    basin_nos = ribasim_model.node.static[ribasim_model.node.static["type"]=="Basin"].index.values
    for basin_no in basin_nos:
        print(f" - Basin {basin_no}")
        plot_results_basin_ribasim_model(
            ribasim_model=ribasim_model,
            simulation_path=simulation_path,
            basin_no=basin_no,
            ribasim_results=ribasim_results,
        );


def plot_results_basin_ribasim_model(
    basin_no: int = None, 
    simulation_path: Path = None, 
    ribasim_model: ribasim.Model = None, 
    ribasim_results: RibasimModelResults = None,
    basin_results: RibasimBasinResults = None
):
    if ribasim_results is None:
        if not simulation_path.exists():
            raise ValueError(" x simulation_path {simulation_path} does not exist")
        ribasim_results = read_ribasim_model_results(simulation_path=simulation_path)
    
    if ribasim_model is None:
        if not simulation_path.exists():
            raise ValueError(" x simulation_path {simulation_path} does not exist")
        ribasim_model = ribasim.Model.from_toml(Path(simulation_path, "ribasim.toml"))
    
    if basin_no is None and basin_results is None:
        raise ValueError(" x no basin_no or basin_results provided")
    if basin_results is None:
        basin_results = get_ribasim_basin_data_from_model(
            ribasim_model=ribasim_model, 
            ribasim_results=ribasim_results, 
            basin_no=basin_no
        )
    if basin_no != basin_results.basin_no:
        basin_no = basin_results.basin_no
    
    basin_name = ribasim_model.node.static.loc[basin_no, "name"]
    if basin_name == "":
        basin_name = "Basin"

    xmin = basin_results.basin.index.min()
    xmax = basin_results.basin.index.max()

    basin_no = basin_results.basin_no
    if basin_results.control_condition is None:
        control_storage = None
        control_level = None
    else:
        control_storage = basin_results.control_condition[basin_results.control_condition['variable'] == 'storage']
        control_level = basin_results.control_condition[basin_results.control_condition['variable'] == 'level']
    
    total_outflow = basin_results.outflow.copy()
    total_inflow = basin_results.inflow.copy()
    basin_flow = basin_results.basin_flow.copy()
    basin = basin_results.basin.merge(basin_flow, how="outer", left_index=True, right_index=True)

    fig = plt.figure(figsize=(14, 7))

    # storage
    ax1 = fig.add_subplot(421)
    ax1.set_title(f"{basin_name} ({basin_no})", fontsize=15)
    # ax1.hlines(
    #     y=basin_results.basin_profile['level'].min(), 
    #     xmin=xmin, xmax=xmax, linestyle="-", color='black'
    # )
    basin.level.interpolate().rename(f"Level").plot(ax=ax1, style="-o", markersize=2)

    # level
    ax2 = fig.add_subplot(423, sharex=ax1)
    ax2.hlines(y=0.0, xmin=xmin, xmax=xmax, linestyle="-", color='black', label=None)
    basin.storage.interpolate().rename(f"Storage").plot(ax=ax2, style="-o", markersize=2)
    if control_storage is not None:
        ax1.hlines(
            y=control_storage.greater_than.values,
            xmin=xmin, xmax=xmax, linestyle="--", color='grey'
        )

    # flow
    ax3 = fig.add_subplot(223, sharex=ax2)
    ax3.hlines(
        y=0.0, xmin=xmin, xmax=xmax, linestyle="-", color='black'
    )
    basin.flow.rename(f"Drainage+infiltration").plot(ax=ax3, style="-o", markersize=2)
    for i, inflow_edge in basin_results.inflow_edge.iterrows():
        from_node_id = inflow_edge["from_node_id"]
        from_node_name = inflow_edge["from_node_name"]
        source_node_id = inflow_edge["source_node_id"]
        source_name = inflow_edge["source_name"]
        (total_inflow.loc[from_node_id]
         .flow.rename(f"Inflow from {source_name} ({source_node_id}) via {from_node_name} ({from_node_id})")
         .plot(ax=ax3, 
            #    drawstyle="steps-post", 
               style="-o", 
               markersize=3))
    for i, outflow_edge in basin_results.outflow_edge.iterrows():
        to_node_id = outflow_edge["to_node_id"]
        to_node_name = outflow_edge["to_node_name"]
        target_node_id = outflow_edge["target_node_id"]
        target_name = outflow_edge["target_name"]
        (total_outflow.loc[to_node_id]
         .flow.rename(f"Outflow to {target_name} ({target_node_id}) via {to_node_name} ({to_node_id})")
         .plot(ax=ax3, 
            #    drawstyle="steps-post", 
               style="-o", 
               markersize=3))

    ax4 = fig.add_subplot(443, sharey=ax1)
    for i, outflow_edge in basin_results.outflow_edge.iterrows():
        to_node_id = outflow_edge["to_node_id"]
        q_h_relation = (basin_results.outflow.loc[to_node_id][["flow"]]
                        .merge(basin_results.basin[["level"]], how="inner", left_index=True, right_index=True))
        ax4.plot(q_h_relation.flow, q_h_relation.level, "o", markersize=3)
    ax4.hlines(
        y=basin_results.basin_profile.level.min(), 
        xmin=0.0, xmax=basin_results.outflow.max().values*1.1, linestyle="-", color='black'
    )
    ax4.set_title(f"Q-H relation", fontsize=10)
    ax4.set_xlim([0.0, basin_results.outflow.max().values*1.1])

    ax5 = fig.add_subplot(444, sharey=ax1)
    basin_profile = pd.concat([
        pd.DataFrame(
            dict(
                level=[basin_results.basin_profile.level.min()], 
                area=[0.0], 
                remarks=[""]
            ), 
            index=[0]
        ), 
        basin_results.basin_profile
    ]).reset_index(drop=True)
    basin_profile.area = basin_profile.area/10_000
    basin_profile.set_index("area").level.rename('H-A relation').plot(ax=ax5, style='o', markersize=3)
    ax5.hlines(
        y=basin_results.basin_profile['level'].min(), 
        xmin=0.0, xmax=basin_profile.area.max()*1.1, linestyle="-", color='black'
    )
    ax5.set_title(f"A-H relation", fontsize=10)
    ax5.set_xlim([0.0, basin_profile.area.max()*1.1])

    # control levels
    if control_level is not None:
        for i, clevel in control_level.iterrows():
            control_node_id = clevel["node_id"]
            controlled_node_id = (ribasim_model.edge.static[
                (ribasim_model.edge.static["edge_type"]=="control") & 
                (ribasim_model.edge.static["from_node_id"]==control_node_id)
            ]["to_node_id"].iloc[0])
            node_name = ribasim_model.node.static.loc[controlled_node_id, "name"]
            ax1.hlines(
                y=clevel["greater_than"], 
                xmin=xmin, xmax=xmax, linestyle="--", color='grey', label=f'Control level {node_name} ({controlled_node_id})'
            )
            ax4.hlines(
                y=clevel["greater_than"], 
                xmin=0.0, 
                xmax=basin_results.outflow.max()*1.1, 
                linestyle="--", 
                color='grey', 
                label=f'Control level {node_name} ({controlled_node_id})'
            )
            ax5.hlines(
                y=clevel["greater_than"], 
                xmin=0.0, 
                xmax=basin_profile.area.max()*1.1, 
                linestyle="--", 
                color='grey', 
                label=f'Control level {node_name} ({controlled_node_id})'
            )

    for ax in [ax1, ax2, ax3]:
        ax.legend(fontsize=8)
        # ax.set_xlim([xmin, xmax])
        # ax.set_xticklabels(ax3.get_xticks(), rotation=0, ha='center')
        # if (xmax-xmin).days < 5.0:
        #     ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d\n %H:%M'))
        # elif (xmax-xmin).days < 90.0:
        #     ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        # elif (xmax-xmin).days < 365.0*5.0:
        #     ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.grid(color='lightgrey', linewidth=0.5)

    ax1.set_ylabel('Level [mAD]')
    ax1.set_xlabel(None)
    ax2.set_ylabel('Storage [m3]')
    ax2.set_xlabel(None)
    ax2.set_xticklabels([])
    ax2.minorticks_off()
    ax3.set_ylabel('Flow [m3/s]')
    ax3.set_xlabel(None)
    ax4.set_ylabel('Level [mAD]')
    ax4.set_xlabel("Flow [m3/s]")
    ax5.set_ylabel('Level [mAD]')
    ax5.set_xlabel("Area [ha]")

    # for ax in [ax1, ax2, ax3, ax4, ax5]:
    #     ax.xaxis.set_tick_params(rotation=0)
    #     ax.yaxis.set_tick_params(labelleft=True, rotation=0)
    #     plt.setp(ax.get_xticklabels(), ha="center", fontsize=8)
    #     plt.setp(ax.get_yticklabels(), fontsize=8)
        
    return ribasim_model, ribasim_results, basin_results, fig, [ax1, ax2, ax3, ax4, ax5]
