"""Polygon list classes and functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from shapely.geometry.polygon import Polygon
from shapely.validation import make_valid

from . import np_box_list


class PolygonList(np_box_list.BoxList):
  """Polygon collection.

  PolygonList represents a list of bounding polygons. A polygon with n vertices
  is represented as a [n, 2] ndarray.

  Optionally, users can add additional related fields (such as
  objectness/classification scores).
  """

  def __init__(self, data, set_aa_box=True):
    """Constructs polygon collection.

    Args:
      data: PolygonList accepts three kinds of representations: (1) a numpy
        array of shape [L, N, 2] for L polygons with the same number of vertices
        (2) a list of ndarray where the i-th ndarray has shape [n_i, 2]. It's
        also stored in the 'boxes' field as a list of Polygon objects. (3) a
        list of Polygon objects
      set_aa_box: Whether to infer axis-aligned bounding boxes from data.

    Raises:
      ValueError: if data format is incorrect.
    """
    if isinstance(data, list):
      for i in range(len(data)):
        if not isinstance(data[i], np.ndarray):
          continue
        shape = data[i].shape
        if len(shape) != 2:
          raise ValueError(f"Shape ({shape}) is not supported for ndarray.")
        n = shape[0]
        if n < 3:
          raise ValueError(f"Invalid number of polygon vertices (N={n}).")
        if shape[1] != 2:
          raise ValueError(f"Invalid polygon formats ({shape}).")
    elif isinstance(data, np.ndarray):
      shape = data.shape
      if len(shape) != 3:
        raise ValueError(f"Shape ({shape}) is not supported for ndarray.")
      _, n, _ = shape
      if n < 3:
        raise ValueError(f"Invalid number of polygon vertices (N={n}).")
      if shape[-1] != 2:
        raise ValueError(f"Invalid polygon formats ({shape}).")
    else:
      raise ValueError(f"Type ({type(data)}) is not supported")

    self.data = {}
    polys = [(Polygon(poly) if isinstance(poly, np.ndarray) else poly) for poly in data]
    self.data["boxes"] = [poly if poly.is_valid else make_valid(poly) for poly in polys]

    # axis_aligned bboxes:
    if set_aa_box:
      self.data["aa_boxes"] = np.array(
          [self.poly_to_axis_aligned_bboxes(poly) for poly in polys])

  def num_boxes(self):
    """Return number of boxes held in collections."""
    return len(self.data["boxes"])

  def get_coordinates(self):
    """This method is not supported for polygon."""
    raise NotImplementedError()

  def poly_to_axis_aligned_bboxes(self, poly):
    """As titled."""
    x, y = poly.exterior.coords.xy
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)
    return [ymin, xmin, ymax, xmax]
