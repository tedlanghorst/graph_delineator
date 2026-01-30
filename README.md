# Graph Delineator

A python tool for creating a customized watershed graph from MERIT-Hydro data. It builds a directed graph where each node represents a MERIT catchment. This graph is then split at locations of gauges (details on the input format below), and then aggregated up to a target catchment size while mantaining these gauges as the outlets of our catchments.


## Installation

I use `uv` for dependency management in this project. First, fork/clone this repo and cd into the root. Then, use `uv sync` to install the dependencies and create virtual environment. 

```bash 
# If you need to, install uv first: 
# pip install uv 
uv sync
```

Then install the kernel so you can run jupyter notebooks
```bash
uv run python -m ipykernel install --user --name graph-delineator --display-name "Graph Delineator (uv)"
```

You should be able to use conda with the requirements listed in `pyproject.toml` if that you prefer conda. 

## Quick Start
The main entry point is the `delineate_basins` function in delineate.py. See the `demo.ipynb` notebook for a quick example.


### Input Requirements
The main input for this code (other than the MERIT data soruces), is a .csv file indicating the gauge that will be targeted for splitting the watershed. This input CSV must have these columns:

| Column | Description |
| --- | ----------- |
| id | Unique gauge identifier |
| lat | Latitude (decimal degrees) |
| lng | Longitude (decimal degrees) |
| COMID | MERIT COMID where this gauge is located |
| position | Fractional distance along the MERIT river line for this gauge [0-1] where 1 represents the downstream end of the line. |
| outlet_id | the unique gauge id for this gauge's farthest downstream outlet. |



### Outputs
Delineation generates two primary datasets per outlet_id in the specified output_dir:

- **<output_dir>/subbasins:**  GeoParquet file containing delineated catchment polygons and attributes (id, area_km2, uparea_km2, nextdown, etc.).

- **<output_dir>/gauges:** GeoParquet file containing the point locations of gauges and their calculated upstream areas.

And, optionally:

- **<output_dir/plots:**> plots of the delineated watersheds and gauges with matching colors for quick evaluation.