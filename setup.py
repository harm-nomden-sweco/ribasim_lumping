from setuptools import setup, find_packages

setup(
    name="ribasim_lumping",
    version="0.0.1",
    author="Harm-Nomden-Sweco",
    author_email="harm.nomden@sweco.nl",
    description="Generate Ribasim Model using lumping/aggregation of d-hydro-simulation",
    packages=find_packages(),
    install_requires=[
        "geopandas==0.12.2",
        "matplotlib==3.7.0",
        "pandas==1.5.3",
        "xarray==2023.2.0",
        "ugrid==0.13.0",
        "xugrid==0.5.0",
        "ribasim==0.2.0",
        "dfm_tools==0.11.0",
        "hydrolib-core==0.5.2",
        "contextily==1.3.0",
        "pydantic==2.0.0",
        "momepy==0.6.0",
        "networkx==3.0",
    ],
)
