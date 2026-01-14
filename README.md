
#  What is this repository for?

ITCOS is an algorithm for optimizing soil erosion monitoring units.

# Environment

python3.10 and an IDE for running the python code.


# How to run the code?

Download all files and run ITCOS.ipynb in an integrated development environment that supports Jupyter files. Ensure all packages used in the code (listed in the first cell of the source code) are properly installed. After configuring the data files in cells 2 and 4, simply click to run automatically.

# Sample data description

`Test_triangular_network.csv` is the triangular network to be optimized for the test, and it already includes the terrain factor (LS). Here, Tri_Index represents the triangle ID, xy denotes the vertex coordinates, and LS is the slope length factor calculated.

`RKCP.tif` is the result obtained from non-terrain factor calculations.

# Reference 

The RT-WFP.ipynb file in this project implements a TIN-based slope length extraction algorithm. This project utilizes this algorithm to calculate slope length factors.
For a detailed introduction to the RT-WFP algorithm, please refer to:https://doi.org/10.1016/j.cageo.2024.105737.

