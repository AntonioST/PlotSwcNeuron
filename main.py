import abc
import argparse
import math
from pathlib import Path
from typing import List, Iterator, Tuple, Dict, Any, overload, Union, ClassVar

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.axes import Axes
from matplotlib.patches import Circle

R2 = Tuple[float, float]


class Vec:
    X: ClassVar
    Y: ClassVar
    Z: ClassVar

    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    @property
    def unit(self) -> 'Vec':
        n = abs(self)
        if n < 1e-5:
            return self  # empty vector

        return Vec(self.x / n, self.y / n, self.z / n)

    def __abs__(self):
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def __add__(self, other: Union[int, float, 'Vec']) -> 'Vec':
        if isinstance(other, (int, float)):
            return Vec(self.x + other, self.y + other, self.z + other)
        else:
            return Vec(self.x + other.x, self.y + other.y, self.z + other.z)

    def __radd__(self, other: Union[int, float]) -> 'Vec':
        if isinstance(other, (int, float)):
            return Vec(self.x + other, self.y + other, self.z + other)
        else:
            raise TypeError()

    def __sub__(self, other: Union[int, float, 'Vec']) -> 'Vec':
        if isinstance(other, (int, float)):
            return Vec(self.x - other, self.y - other, self.z - other)
        else:
            return Vec(self.x - other.x, self.y - other.y, self.z - other.z)

    def __rsub__(self, other: Union[int, float]) -> 'Vec':
        if isinstance(other, (int, float)):
            return Vec(other - self.x, other - self.y, other - self.z)
        else:
            raise TypeError()

    @overload
    def __mul__(self, other: Union[int, float]) -> 'Vec':
        pass

    @overload
    def __mul__(self, other: 'Vec') -> float:
        pass

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Vec(self.x * other, self.y * other, self.z * other)
        else:
            return self.x * other.x + self.y * other.y + self.z * other.z

    def __rmul__(self, other: Union[int, float]):
        if isinstance(other, (int, float)):
            return Vec(self.x * other, self.y * other, self.z * other)
        else:
            raise TypeError()

    def __truediv__(self, other: Union[int, float, 'Vec']) -> 'Vec':
        if isinstance(other, (int, float)):
            return Vec(self.x / other, self.y / other, self.z / other)
        else:
            # project
            q = other.unit
            return self * q * q

    def __matmul__(self, other: 'Vec') -> 'Vec':
        return Vec(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def __str__(self):
        return f'({self.x}, {self.y}, {self.z})'

    __repr__ = __str__

    def __hash__(self):
        return hash((self.x, self.y, self.z))

    def __eq__(self, other):
        try:
            return self.x == other.x and self.y == other.y and self.z == other.z
        except AttributeError:
            return False


Vec.X = Vec(1, 0, 0)
Vec.Y = Vec(0, 1, 0)
Vec.Z = Vec(0, 0, 1)


class SwcNode:
    node_number: int
    identifier: int
    """
    Standardized swc files (www.neuromorpho.org) - 
    0 - undefined
    1 - soma
    2 - axon
    3 - (basal) dendrite
    4 - apical dendrite
    5+ - custom
    """

    vec: Vec
    r: float
    parent: int

    def __init__(self, n: int, s: int, x: float, y: float, z: float, r: float, p: int):
        self.node_number = n
        self.identifier = s
        self.vec = Vec(x, y, z)
        self.r = r
        self.parent = p

    @property
    def x(self) -> float:
        return self.vec.x

    @property
    def y(self) -> float:
        return self.vec.y

    @property
    def z(self) -> float:
        return self.vec.z

    @property
    def is_undefined(self) -> bool:
        return self.identifier == 0

    @property
    def is_soma(self) -> bool:
        return self.identifier == 1

    @property
    def is_axon(self) -> bool:
        return self.identifier == 2

    @property
    def is_dendrite(self) -> bool:
        return self.identifier == 3 or self.identifier == 4

    def __str__(self):  # todo comment
        return ' '.join(map(str, [
            self.node_number,
            self.identifier,
            self.x,
            self.y,
            self.z,
            self.r,
            self.parent
        ]))

    def __repr__(self):  # todo comment
        return f'{self.node_number}, ' \
               f'id={self.identifier} ' \
               f'p=({self.x}, {self.y}, {self.z}), ' \
               f'r={self.r}, ' \
               f'parent={self.parent}'


class Swc:
    node: List[SwcNode]

    def __init__(self, node: List[SwcNode]):
        self.node = node

    @classmethod
    def load(cls, file: Path) -> 'Swc':  # TODO comment
        node = []

        # UnicodeDecodeError: 'utf-8' codec can't decode byte 0xa4 in position 205: invalid start byte
        # If an error happened like above, please change the encoding below, you can try those encoding
        # to load the swc file
        #   utf-8 (default)
        #   Big5
        #

        with file.open('r', encoding='Big5') as f:
            for line in f:
                line = line.strip()

                if len(line) == 0:
                    continue
                if line.startswith('#'):
                    continue

                part = line.split()
                n = int(part[0])
                s = int(part[1])
                x = float(part[2])
                y = float(part[3])
                z = float(part[4])
                r = float(part[5])
                p = int(part[6])

                node.append(SwcNode(n, s, x, y, z, r, p))

        return Swc(node)

    def __getitem__(self, item: int) -> SwcNode:
        """Get SwcNode whose node_number equals to item.

        :param item: node_number value
        :return: found node
        :raise IndexError: node found found
        """
        try:
            ret = self.node[item - 1]
        except IndexError:
            ret = None

        if ret is not None and ret.node_number == item:
            return ret

        for node in self.node:
            if node.node_number == item:
                return node

        raise IndexError

    def foreach_node(self) -> Iterator[SwcNode]:
        """Foreach nodes"""
        for node in self.node:
            yield node

    def foreach_line(self) -> Iterator[Tuple[SwcNode, SwcNode]]:
        """Foreach line segments

        :return: tuple of 2 nodes (child, parent).
        """
        for node in self.node:
            if node.parent > 0:
                yield node, self[node.parent]

    def __str__(self):
        line = []
        for node in self.node:
            line.append(str(node))
        return '\n'.join(line)


class Projection(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __call__(self, p: Vec) -> R2:
        pass

    @classmethod
    def new(cls, expr: str) -> 'Projection':
        if ':' in expr:
            i = expr.index(':')
            name = expr[:i]
            expr = expr[i + 1:]
        else:
            name = expr
            expr = ''

        try:
            proj_type = globals()[name + 'Projection']
        except KeyError as e:
            raise RuntimeError(f'Projection : {name}Projection not found') from e

        if not issubclass(proj_type, Projection):
            raise TypeError(f'not a Projection : {proj_type}')

        return proj_type.new(expr)


class DefaultProjection(Projection):

    def __call__(self, p: Vec) -> R2:
        return p.x, p.y

    @classmethod
    def new(cls, expr: str) -> 'DefaultProjection':
        return DefaultProjection()


class OrthographicProjection(Projection):
    def __init__(self, v: Vec, t: Vec = Vec.Z, z_factor: float = 1.0):
        self.u = v.unit
        self.t = t.unit
        self.z = z_factor

    def __call__(self, p: Vec) -> R2:
        u = self.u
        q = Vec(p.x, p.y, p.z * self.z)
        a = q / u
        b = q - a
        cx = b / self.t
        cy = b - cx
        return abs(cx), abs(cy)

    @classmethod
    def new(cls, expr: str) -> 'OrthographicProjection':
        pattern = "X,Y,Z[,X',Y',Z'][*F]"
        part = expr.split(',')

        p = [Vec.Z, Vec.Z]
        z = 1.0
        if len(part) > 0:
            s = 0
            i = 0

            while i < len(part):
                try:
                    if part[i].startswith('*'):
                        z = float(part[i][1:])
                        i += 1
                    elif part[i] in 'X':
                        p[s] = Vec.X
                        i += 1
                        s += 1
                    elif part[i] in 'Y':
                        p[s] = Vec.Y
                        i += 1
                        s += 1
                    elif part[i] in 'Z':
                        p[s] = Vec.Z
                        i += 1
                        s += 1
                    else:
                        p[s] = Vec(float(part[i]), float(part[i + 1]), float(part[i + 2])).unit
                        i += 3
                        s += 1

                except (IndexError, ValueError) as e:
                    raise ValueError(f'illegal pattern {pattern} : {expr}') from e

        if s == 1 and p[0] * p[1] == 1:
            if p[0] == Vec.Z:
                p[1] = Vec.Y

        return OrthographicProjection(p[0], p[1], z)


class SegStyle(metaclass=abc.ABCMeta):
    def setup(self, swc: Swc):
        pass

    @abc.abstractmethod
    def __call__(self, n1: SwcNode, n2: SwcNode, **kwargs) -> Dict[str, Any]:
        pass

    @classmethod
    def new(cls, expr: str) -> 'SegStyle':
        if ':' in expr:
            i = expr.index(':')
            name = expr[:i]
            expr = expr[i + 1:]
        else:
            name = expr
            expr = ''

        try:
            seg_type = globals()[name + 'SegStyle']
        except KeyError as e:
            raise RuntimeError(f'SegStyle : {name}SegStyle not found')

        if not issubclass(seg_type, SegStyle):
            raise TypeError(f'not a SegStyle : {seg_type}')

        return seg_type.new(expr)


class DefaultSegStyle(SegStyle):

    @classmethod
    def new(cls, expr: str) -> 'DefaultSegStyle':
        return DefaultSegStyle()

    def __call__(self, n1: SwcNode, n2: SwcNode, **kwargs) -> Dict[str, Any]:
        return kwargs


class ColorZSegStyle(SegStyle):
    z: Tuple[float, float]

    def __init__(self, code: str, rg: Tuple[float, float]):
        if code == 'h':
            self.color_change = self.color_change_Hsv
        elif code == 's':
            self.color_change = self.color_change_hSv
        elif code == 'v':
            self.color_change = self.color_change_hsV
        else:
            raise ValueError()

        if not (0 <= rg[0] < rg[1] <= 1):
            raise ValueError(f'illegal range : {rg}')

        self.range = rg[1] - rg[0], rg[0]

    @classmethod
    def new(cls, expr: str) -> 'ColorZSegStyle':
        pattern = '[HSV][A~B]'
        if expr.startswith('H'):
            code = 'h'
        elif expr.startswith('S'):
            code = 's'
        elif expr.startswith('V'):
            code = 'v'
        else:
            raise ValueError(f'illegal pattern {pattern} : {expr}')

        try:
            expr = expr[1:]
            if '~' not in expr:
                lo = float(expr)
                hi = 1.0
            else:
                i = expr.index('~')
                lo = float(expr[:i])
                hi = float(expr[i + 1:])
        except (IndexError, ValueError) as e:
            raise ValueError(f'illegal pattern {pattern} : {expr}')

        return ColorZSegStyle(code, (lo, hi))

    def setup(self, swc: Swc):
        z = np.array([n.z for n in swc.node])
        z = np.max(z), np.min(z)
        self.z = z[0], z[1] - z[0]

    def __call__(self, n1: SwcNode, n2: SwcNode, **kwargs) -> Dict[str, Any]:
        try:
            c = kwargs['color']
        except KeyError:
            pass
        else:
            z = (n1.z + n2.z) / 2  # average z
            z = (z - self.z[0]) / self.z[1]  # normalize [0, 1]
            a = z * self.range[0] - self.range[1]
            kwargs['color'] = self.color_change(c, a)

        return kwargs

    # noinspection PyPep8Naming
    @staticmethod
    def color_change_hsV(color: str, f: float):
        h, s, v = colors.rgb_to_hsv(ColorZSegStyle.get_color_rgb_code(color))
        v = max(1, min(0, v * f))
        return colors.hsv_to_rgb((h, s, v))

    # noinspection PyPep8Naming
    @staticmethod
    def color_change_hSv(color: str, f: float):
        h, s, v = colors.rgb_to_hsv(ColorZSegStyle.get_color_rgb_code(color))
        s = max(1, min(0, s * f))
        return colors.hsv_to_rgb((h, s, v))

    # noinspection PyPep8Naming
    @staticmethod
    def color_change_Hsv(color: str, f: float):
        h, s, v = colors.rgb_to_hsv(ColorZSegStyle.get_color_rgb_code(color))
        h = (h + f + 1) % 1
        return colors.hsv_to_rgb((h, s, v))

    @staticmethod
    def get_color_rgb_code(color):
        try:
            c = colors.BASE_COLORS[color]
        except KeyError:
            try:
                c = colors.cnames[color]
            except KeyError:
                c = color
        return c


def smooth_line_radius(ax: Axes,
                       p1: Tuple[float, float], w1: float,
                       p2: Tuple[float, float], w2: float,
                       num=2,
                       **kwargs):
    """plot smoothed line.

    :param ax:
    :param p1: point1
    :param w1: width at point1
    :param p2: point2
    :param w2: width at point2
    :param num: number of segments.
    :param kwargs: other line property
    """
    px = np.linspace(p1[0], p2[0], num + 1)
    py = np.linspace(p1[1], p2[1], num + 1)
    lw = np.linspace(w1, w2, num)
    for i in range(num):
        ax.plot(px[i:i + 2], py[i:i + 2], lw=lw[i], **kwargs)


# change color for each segment
DEFAULT_COLOR = {
    'body': 'k',
    'soma': 'b',
    'axon': 'r',
    'dendrite': 'g'
}


def plot_swc(ax: Axes, swc: Swc,
             radius=True,
             color: Dict[str, str] = None,
             projection: Projection = None,
             segment_style: SegStyle = None):
    """Plot swc neuron.

    :param ax:
    :param swc:
    :param radius: plot radius
    :param color: a dictionary of terminal color.
    :param projection: projection method, callable of `(3D_point) -> 2D_point`
    """
    if color is None:
        color = DEFAULT_COLOR

    if projection is None:
        projection = DefaultProjection()

    if segment_style is None:
        segment_style = DefaultSegStyle()

    segment_style.setup(swc)

    for n1, n2 in swc.foreach_line():
        p1 = projection(n1.vec)
        p2 = projection(n2.vec)

        if n1.is_soma:
            c = color['soma']
        elif n1.is_axon:
            c = color['axon']
        elif n1.is_dendrite:
            c = color['dendrite']
        else:
            c = color['body']

        k = segment_style(n1, n2, color=c, solid_capstyle='round')

        if radius:
            if n2.is_soma:
                ax.add_artist(Circle(p2, n2.r, color=color['soma']))
                if not n1.is_soma:
                    smooth_line_radius(ax, p1, n1.r, p2, n1.r, **k)
            else:
                smooth_line_radius(ax, p1, n1.r, p2, n2.r, **k)
        else:
            px = p1[0], p2[0]
            py = p1[1], p2[1]
            ax.plot(px, py, **k)


def main(args: List[str] = None):
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--output',
                    default=None,
                    help='output picture name',
                    dest='output')
    ap.add_argument('-r', '--use-radius',
                    action='store_true',
                    help='plot radius',
                    dest='use_radius')
    ap.add_argument('-P', '--projection',
                    metavar='CLASS[:EXPR]',
                    default=None,
                    help='projection method',
                    dest='projection')
    ap.add_argument('-S', '--segment-style',
                    metavar='CLASS[:EXPR]',
                    default=None,
                    help='segment style',
                    dest='segment_style')
    ap.add_argument('--show',
                    action='store_true',
                    help='show preview')
    ap.add_argument('FILE',
                    help='swc file')

    opt = ap.parse_args(args)
    swc = opt.FILE
    swc = Swc.load(Path(swc))

    proj = opt.projection
    if proj is not None:
        proj = Projection.new(proj)

    seg = opt.segment_style
    if seg is not None:
        seg = SegStyle.new(seg)

    fig, ax = plt.subplots()
    plot_swc(ax, swc, radius=opt.use_radius, projection=proj, segment_style=seg)
    ax.axis('equal')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.tight_layout()

    if opt.show:
        plt.show()
    else:
        output = opt.output
        if output is None:
            output = opt.FILE.replace('.swc', '.png')  # output .png if there is no given output
        plt.savefig(output, dpi=300)
        plt.clf()


if __name__ == '__main__':
    main()
