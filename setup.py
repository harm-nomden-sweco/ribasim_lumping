from setuptools import setup, find_packages

setup(
    name="ribasim_lumping",
    version="0.1.0",
    author="Harm-Nomden-Sweco",
    author_email="harm.nomden@sweco.nl",
    description="Generate Ribasim Model using lumping/aggregation of d-hydro-simulation",
    packages=find_packages(),
    install_requires=[
	"geopandas==0.14.1",
	"matplotlib==3.8.2",
	"xarray==2023.11.0",
	"ugrid==0.13.0",
	"xugrid==0.7.1",
	"pandera==0.17.2",
	"tomli==2.0.1",
	"tomli_w==1.0.0",
	"networkx==3.2.1",
	"black==23.11.0",
	"openpyxl",
	"ipykernel",
	"pandas-xlsx-tables==1.0.0",
	"pyarrow==14.0.1",
	"dfm_tools==0.18.0"
    ],
)
