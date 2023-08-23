# ribasim-lumping

This python package is used to develop an aggregated Ribasim network. It was developed by Sweco (Harm Nomden and Tessa Andringa) when working on the TKI-project (programme NHI) on the development, application, and testing of the new Ribasim-model (Deltares, https://github.com/Deltares/Ribasim). 

### Objective
It translates a D-Hydro network into an aggregated Ribasim-network. The detailed D-Hydro network is divided into 'basins' based on locations provided by the user where the network should be split. On these split-locations nodes are placed which define the exchange flow between the basins.

### Dependencies
Most important dependencies:
- NETWORKX (https://networkx.org/, Hagberg et al., 2008)
- UGRID (https://github.com/Deltares/UGridPy)
- XUGRID (https://github.com/Deltares/xugrid)

### Installation
We will make this package accessible via pypi. It is recommended to clone this repository because it is under development and it includes some example notebooks.

### Licences
This package is developed under the MIT license. Reference to this package: Ribasim-Lumping (Sweco, 2023).

### References:
Aric A. Hagberg, Daniel A. Schult and Pieter J. Swart, “Exploring network structure, dynamics, and function using NetworkX”, in Proceedings of the 7th Python in Science Conference (SciPy2008), Gäel Varoquaux, Travis Vaught, and Jarrod Millman (Eds), (Pasadena, CA USA), pp. 11–15, Aug 2008
