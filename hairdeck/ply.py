"""PLY point clouds in the layout of the CHARM evaluation protocol, read and written with numpy only.

CHARM's sample.py writes its point clouds with open3d: binary little endian, one vertex element with the properties
double x y z nx ny nz and uchar red green blue (all CHARM-protocol point clouds of the release have this layout).
read: the vertex element of any binary little-endian PLY file whose first element is the vertex element with scalar
properties, as a numpy structured array; read_points: its x y z as float64 (N, 3), the values open3d's reader gives.
write: that layout (grey colours, as CHARM's sample.py writes them); open3d reads the files back unchanged.
"""
import numpy as np

TYPES = {'char': 'i1', 'int8': 'i1', 'uchar': 'u1', 'uint8': 'u1', 'short': '<i2', 'int16': '<i2', 'ushort': '<u2',
         'uint16': '<u2', 'int': '<i4', 'int32': '<i4', 'uint': '<u4', 'uint32': '<u4', 'float': '<f4', 'float32': '<f4',
         'double': '<f8', 'float64': '<f8'}


def read(path):
    """The vertex element of a binary little-endian PLY file -> structured array (one field per property)."""
    with open(path, 'rb') as f:
        raw = f.read()
    end = raw.index(b'end_header')
    end = raw.index(b'\n', end) + 1
    lines = [l.strip() for l in raw[:end].decode('ascii').splitlines()]
    if lines[0] != 'ply' or 'format binary_little_endian 1.0' not in lines:
        raise ValueError(f'{path}: not a binary little-endian PLY file')
    elements = [l.split() for l in lines if l.startswith('element ')]
    if not elements or elements[0][1] != 'vertex':
        raise ValueError(f'{path}: the first element is not the vertex element')
    n, fields, inside = int(elements[0][2]), [], False
    for l in lines:
        if l.startswith('element '):
            inside = l.split()[1] == 'vertex'
        elif l.startswith('property ') and inside:
            p = l.split()
            if p[1] == 'list':
                raise ValueError(f'{path}: list property in the vertex element')
            fields.append((p[2], TYPES[p[1]]))
    dt = np.dtype(fields)
    return np.frombuffer(raw[end:end + n * dt.itemsize], dtype=dt).copy()


def read_points(path):
    """x y z of the vertex element as float64 (N, 3)."""
    v = read(path)
    return np.stack([v['x'], v['y'], v['z']], 1).astype(np.float64)


HEADER = ('ply\nformat binary_little_endian 1.0\nelement vertex {n}\nproperty double x\nproperty double y\nproperty double z\n'
          'property double nx\nproperty double ny\nproperty double nz\nproperty uchar red\nproperty uchar green\n'
          'property uchar blue\nend_header\n')


def write(path, points, normals, grey=128):
    """points, normals (N, 3) -> PLY in CHARM's layout (double positions and normals, uchar colours all grey, the 0.5 of
    CHARM's sample.py)."""
    dt = np.dtype([('p', '<f8', 3), ('n', '<f8', 3), ('c', 'u1', 3)])
    v = np.empty(len(points), dt)
    v['p'], v['n'], v['c'] = points, normals, grey
    with open(path, 'wb') as f:
        f.write(HEADER.format(n=len(points)).encode('ascii'))
        f.write(v.tobytes())
