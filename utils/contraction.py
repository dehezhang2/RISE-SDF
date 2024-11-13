"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

from enum import Enum


class ContractionType(Enum):
    """Space contraction options.

    This is an enum class that describes how a :class:`nerfacc.Grid` covers the 3D space.
    It is also used by :func:`nerfacc.ray_marching` to determine how to perform ray marching
    within the grid.

    The options in this enum class are:

    Attributes:
        AABB: Linearly map the region of interest :math:`[x_0, x_1]` to a
            unit cube in :math:`[0, 1]`.

            .. math:: f(x) = \\frac{x - x_0}{x_1 - x_0}

        UN_BOUNDED_TANH: Contract an unbounded space into a unit cube in :math:`[0, 1]`
            using tanh. The region of interest :math:`[x_0, x_1]` is first
            mapped into :math:`[-0.5, +0.5]` before applying tanh.

            .. math:: f(x) = \\frac{1}{2}(tanh(\\frac{x - x_0}{x_1 - x_0} - \\frac{1}{2}) + 1)

        UN_BOUNDED_SPHERE: Contract an unbounded space into a unit sphere. Used in
            `Mip-Nerf 360: Unbounded Anti-Aliased Neural Radiance Fields`_.

            .. math::
                f(x) =
                \\begin{cases}
                z(x) & ||z(x)|| \\leq 1 \\\\
                (2 - \\frac{1}{||z(x)||})(\\frac{z(x)}{||z(x)||}) & ||z(x)|| > 1
                \\end{cases}

            .. math::
                z(x) = \\frac{x - x_0}{x_1 - x_0} * 2 - 1

            .. _Mip-Nerf 360\: Unbounded Anti-Aliased Neural Radiance Fields:
                https://arxiv.org/abs/2111.12077

    """

    AABB = 0
    UN_BOUNDED_TANH = 1
    UN_BOUNDED_SPHERE = 2
