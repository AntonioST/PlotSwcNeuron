import argparse
from pathlib import Path
from typing import List, Iterator, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Circle


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

    x: float
    y: float
    z: float
    r: float
    parent: int

    def __init__(self, n: int, s: int, x: float, y: float, z: float, r: float, p: int):
        self.node_number = n
        self.identifier = s
        self.x = x
        self.y = y
        self.z = z
        self.r = r
        self.parent = p

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

    def __str__(self):
        return ' '.join(map(str, [
            self.node_number,
            self.identifier,
            self.x,
            self.y,
            self.z,
            self.r,
            self.parent
        ]))

    def __repr__(self):
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
    def load(cls, file: Path) -> 'Swc':
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
        usage: self[node.parent]

        :param item: node_number value
        :return: found node
        :raise IndexError: node found found
        """
        # fast path
        try:
            ret = self.node[item - 1]
        except IndexError:
            ret = None

        if ret is not None and ret.node_number == item:
            return ret

        # slow path
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
                yield node, self[node.parent]  # getitem

    def __str__(self):
        line = []
        for node in self.node:
            line.append(str(node))
        return '\n'.join(line)


def default_project(p: Tuple[float, float, float]) -> Tuple[float, float]:
    """Default projection function, remove z value.

    :param p: 3d points
    :return:  2d points
    """
    return p[0], p[1]


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
             projection=default_project):
    """Plot swc neuron.

    :param ax:
    :param swc:
    :param radius: plot radius
    :param color: a dictionary of terminal color.
    :param projection: projection method, callable of `(3D_point) -> 2D_point`
    """
    if color is None:
        color = DEFAULT_COLOR

    for n1, n2 in swc.foreach_line():
        p1 = projection((n1.x, n1.y, n1.z))
        p2 = projection((n2.x, n2.y, n2.z))

        if n1.is_soma:
            c = color['soma']
        elif n1.is_axon:
            c = color['axon']
        elif n1.is_dendrite:
            c = color['dendrite']
        else:
            c = color['body']

        if radius:
            if n2.is_soma:
                ax.add_artist(Circle(p2, n2.r, color=color['soma']))
                if not n1.is_soma:
                    smooth_line_radius(ax, p1, n1.r, p2, n1.r, color=c, solid_capstyle='round')
            else:
                smooth_line_radius(ax, p1, n1.r, p2, n2.r, color=c, solid_capstyle='round')
        else:
            px = p1[0], p2[0]
            py = p1[1], p2[1]
            ax.plot(px, py, color=c, solid_capstyle='round')


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
    ap.add_argument('--show',
                    action='store_true',
                    help='show preview')
    ap.add_argument('FILE',
                    help='swc file')

    opt = ap.parse_args(args)
    swc = opt.FILE
    swc = Swc.load(Path(swc))

    fig, ax = plt.subplots()
    plot_swc(ax, swc, radius=opt.use_radius)
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
