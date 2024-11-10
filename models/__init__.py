models = {}


def register(name):
    def decorator(cls):
        models[name] = cls
        return cls

    return decorator


def make(name, config):
    model = models[name](config)
    return model

from . import (
    neus,
    split_mixed_occ,
    geometry,
    texture,
)
# Import external PBR classes
import lib.pbr

EnvironmentLightMipCube = register("envlight-mip-cube")(lib.pbr.EnvironmentLightMipCube)
