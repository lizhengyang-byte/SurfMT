"""SurfMT-LightGBM: LightGBM multi-task version of SurfMT-GNN.

Uses tabular features (ECFP4 fingerprint + 12 RDKit descriptors + temperature)
and trains one LightGBM regressor per task. Handles missing labels by training
each task only on its labeled samples.
"""