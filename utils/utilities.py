import collections
from typing import List, Optional, Union


class DisjointSet:
    """
    A disjoint set implementation from HierText
    github.com/tensorflow/models/blob/master/official/projects/unified_detector/utils/utilities.py
    """

    def __init__(self, num_elements: int):
        self._num_elements = num_elements
        self._parent = list(range(num_elements))

    def find(self, item: int) -> int:
        if self._parent[item] == item:
          return item
        else:
          self._parent[item] = self.find(self._parent[item])
          return self._parent[item]

    def union(self, i1: int, i2: int) -> None:
        r1 = self.find(i1)
        r2 = self.find(i2)
        self._parent[r1] = r2

    def to_group(self) -> List[List[int]]:
        """Return the grouping results.

        Returns:
            A list of integer lists. Each list represents the IDs belonging to the
          same group.
        """
        groups = collections.defaultdict(list)
        for i in range(self._num_elements):
          r = self.find(i)
          groups[r].append(i)
        return list(groups.values())
