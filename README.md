# LNG Maritime Network Analysis

This repository contains the code and supporting geographic files developed for a master's thesis on the criticality of terminals and maritime chokepoints in the global LNG trade network (2020–2024).

## Chokepoint geometry files

* `PortWatch_28_geometry_registry.xlsx` documents the sources, reference information and geometry-processing decisions for the 28 maritime chokepoints included in the analysis.
* `PortWatch_28_chokepoints_geometry.geojson` contains the final polygon geometries used to identify which chokepoints are crossed by the reconstructed LNG routes.

These files are used by Script 01, which prepares the matched LNG voyages and the node table required by the subsequent network-construction pipeline. The original LNG-T3 input files are not redistributed in this repository.

## Methodological overview

Unique origin–destination routes are reconstructed from LNG voyage data and intersected with the final chokepoint geometries. The resulting sequence of terminals and chokepoints is then used to construct a directed monthly multilayer network and to analyse node criticality through measures of scale, structural position, concentration and dependence.

## Data sources

The principal sources are the `LNG_tanker_voyage.csv` and `LNG_terminal.csv` files from the LNG-T3 dataset (Zhou et al., 2026), IMF PortWatch, and the additional geographic references documented in the geometry registry.
