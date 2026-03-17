"""
Datasets package - exposes all dataset classes for easy import.
Usage: from datasets import Shapes3DDataset, VAEDataset
"""

__all__ = []

# Import Shapes3DDataset
try:
    from .Shapes3DDataset import Shapes3DDataset
    __all__.append('Shapes3DDataset')
except Exception as e:
    Shapes3DDataset = None
    print(f"Warning: Could not import Shapes3DDataset: {e}")

# Import VAEDataset from shapes.py
try:
    from .shapes import VAEDataset
    __all__.append('VAEDataset')
except Exception as e:
    VAEDataset = None
    print(f"Warning: Could not import VAEDataset: {e}")

# Add any other dataset classes here following the same pattern
# try:
#     from .your_dataset import YourDataset
#     __all__.append('YourDataset')
# except Exception as e:
#     YourDataset = None
#     print(f"Warning: Could not import YourDataset: {e}")
