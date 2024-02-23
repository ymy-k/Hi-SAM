"""Operations for polygon lists."""

from absl import logging
import numpy as np

from . import np_box_list
from . import np_box_list_ops
from . import polygon_list


EPSILON = 1e-5


def area(polygons: polygon_list.PolygonList):
  """Computes area of masks.

  Args:
    polygons: A polygon_list.PolygonList object.

  Returns:
    a numpy array with shape [N*1] representing polygons areas.
  """
  n = polygons.num_boxes()
  polygon_data = polygons.get()
  return np.array([polygon_data[i].area for i in range(n)])


def intersection_and_area(polygons1, polygons2):
  """Computes area of masks.

  Args:
    polygons1: A polygon_list.PolygonList object.
    polygons2: A polygon_list.PolygonList object.

  Returns:
    a numpy array with shape [N, M] representing intersections areas.
    a numpy array with shape [N, 1] representing the area of polygons1.
    a numpy array with shape [1, M] representing the area of polygons2.
  """
  n = polygons1.num_boxes()
  m = polygons2.num_boxes()
  polygon_data1 = polygons1.get()
  polygon_data2 = polygons2.get()
  intersection_area = np.zeros((n, m), dtype=np.float)
  area1 = area(polygons1)
  area2 = area(polygons2)

  # Use axis-aligned bboxes for fast filtering.
  aa_box_1 = np_box_list.BoxList(polygons1.get_field('aa_boxes'))
  aa_box_2 = np_box_list.BoxList(polygons2.get_field('aa_boxes'))
  intersection_aa_box = np_box_list_ops.intersection(aa_box_1, aa_box_2)
  for i in range(n):
    for j in range(m):
      if intersection_aa_box[i, j] > 0:
        if area1[i] < EPSILON or area2[j] < EPSILON:
          if area1[i] < EPSILON:
            logging.error(  # pylint: disable=logging-format-interpolation
                f'{i}-th text box in array 1 is empty: {polygon_data1[i]}')
          if area2[j] < EPSILON:
            logging.error(  # pylint: disable=logging-format-interpolation
                f'{j}-th text box in array 2 is empty: {polygon_data2[j]}')
          intersection_area[i, j] = 0.
        else:
          intersection_area[i, j] = polygon_data1[i].intersection(
              polygon_data2[j]).area
      else:
        intersection_area[i, j] = 0.
  return intersection_area, area1.reshape(-1, 1), area2.reshape(1, -1)


def iou(polygons1, polygons2):
  """Computes area of masks.

  Args:
    polygons1: A polygon_list.PolygonList object.
    polygons2: A polygon_list.PolygonList object.

  Returns:
    a numpy array with shape [N, M] representing IoU.
  """
  intersection_area, area1, area2 = intersection_and_area(
      polygons1, polygons2)
  union_area = area1 + area2 - intersection_area
  return intersection_area / (union_area + EPSILON)


def ioa(polygons1, polygons2):
  """Computes area of masks.

  Args:
    polygons1: A polygon_list.PolygonList object.
    polygons2: A polygon_list.PolygonList object.

  Returns:
    a numpy array with shape [N, M] representing intersection over the area of
    polygons2.
  """
  intersection_area, _, area2 = intersection_and_area(polygons1, polygons2)
  return intersection_area / (area2 + EPSILON)


def gather(polygons, indices, fields=None):
  """Gather boxes from polygons according to indices and return new polygons.

  By default, gather returns PolygonList corresponding to the input index list,
  as well as all additional fields stored in the PolygonList (indexing into the
  first dimension).  However one can optionally only gather from a
  subset of fields.

  Args:
    polygons: PolygonList holding N boxes
    indices: a 1-d numpy array of type int_
    fields: (optional) list of fields to also gather from.  If None (default),
        all fields are gathered from.  Pass an empty fields list to only gather
        the box coordinates.

  Returns:
    subboxlist: a PolygonList corresponding to the subset of the input
      PolygonList specified by indices

  Raises:
    ValueError: if specified field is not contained in boxlist or if the
        indices are not of type int_
  """
  if indices.size:
    if (np.amax(indices) >= polygons.num_boxes() or
        np.amin(indices) < 0):
      raise ValueError('indices are out of valid range.')
  polygon_data = polygons.get()
  sublist_polygon_data = polygon_list.PolygonList(
      [polygon_data[index] for index in indices],
      set_aa_box=False,
      )
  if fields is None:
    fields = polygons.get_extra_fields()
  for field in fields:
    extra_field_data = polygons.get_field(field)
    sublist_polygon_data.add_field(field, extra_field_data[indices, ...])
  return sublist_polygon_data

