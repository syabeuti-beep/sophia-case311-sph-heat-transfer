# Case 3-1-1 input generation report

DEM particles: 320000
Gas SPH particles: 40000
Total: 360000
Domain: {'x': [0.0, 0.2], 'y': [0.0, 0.04], 'z': [0.0, 0.04]}

The original paper uses an Eulerian grid heat-transfer solve. This case keeps the fixed-bed geometry but replaces solid heat conduction with DEM-neighbor SPH interpolation in function_SPH_DEM_COUPLING.cuh.
