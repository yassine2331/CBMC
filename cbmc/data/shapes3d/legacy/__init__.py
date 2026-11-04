"""
Original 3dshapes loaders copied from the old CBMC-test repo, kept as-is for
reference and for notebooks/3dshapes_experiment.ipynb.

    legacy/3dshapes.py      H5DataSource / Shapes3DDataModule (h5 reader)
    legacy/shapes.py        VAEDataset (folder + metadata.csv, needs pytorch_lightning)
    legacy/extractH5data.py dump 3dshapes.h5 to images/ + metadata.csv

New code should use `cbmc.data.shapes3d` instead.
"""
