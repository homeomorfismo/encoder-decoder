"""
Basic 2D geometries.
"""

from ngsolve import Draw, Mesh
from netgen.geom2d import SplineGeometry


def make_unit_square() -> SplineGeometry:
    """
    Create a unit square mesh.
    """
    geo = SplineGeometry()
    points = [(0, 0), (1, 0), (1, 1), (0, 1)]
    points_ids = [geo.AppendPoint(*point) for point in points]
    lines = [
        [["line", points_ids[i % 4], points_ids[(i + 1) % 4]], "boundary"]
        for i in range(4)
    ]
    for line, bc in lines:
        geo.Append(line, bc=bc)
    return geo


def make_l_shape() -> SplineGeometry:
    """
    Create an L-shaped domain mesh.
    """
    geo = SplineGeometry()
    points = [(0, 0), (0, 1), (-1, 1), (-1, -1), (1, -1), (1, 0)]
    points_ids = [geo.AppendPoint(*point) for point in points]
    lines = [
        [["line", points_ids[i % 6], points_ids[(i + 1) % 6]], "boundary"]
        for i in range(6)
    ]
    for line, bc in lines:
        geo.Append(line, bc=bc)
    return geo


if __name__ == "__main__":
    Draw(make_unit_square())
    input("Press any key to continue...")
    Draw(make_l_shape())
    input("Press any key to continue...")
