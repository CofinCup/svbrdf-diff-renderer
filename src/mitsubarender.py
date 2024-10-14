import numpy as np
import mitsuba as mi  # pip install mitsuba
mi.set_variant("scalar_rgb")
from mitsuba import ScalarTransform4f as T


class MitsubaRender:
    def __init__(self, im_size, envmap_dir, maps_dir):
        self.scene = self.scene_dict()
        self.scene['sensor']['film']['width'] = im_size[0]
        self.scene['sensor']['film']['height'] = im_size[1]
        self.load_texturemaps(maps_dir)

    def render(self, light_pos, light_pow, cam_pos):
        if light_pos is None:
            print("[Warning] light_pos is None, using default light position")
        self.scene['light']['position'] = light_pos
        self.scene['light']['intensity']['value'] = light_pow
        self.scene['sensor']['to_world'] = T.look_at(
            target=[0, 0, 0],
            origin=cam_pos,
            up=[0, 1, 0])
        self.scene['shape']['to_world'] = \
                    T.translate([0, 0, 0]).rotate([1, 0, 0], -30).rotate([0, 1, 0], 0).rotate([0, 0, 1], -10).scale([0.3, 0.3, 0.3])
        return np.array(mi.render(mi.load_dict(self.scene))).clip(0, 1) ** (1 / 2.2)

    def load_texturemaps(self, maps_dir):
        self.scene['shape']['bsdf']['normalmap']['filename'] = maps_dir+'/nom.png'
        self.scene['shape']['bsdf']['bsdf']['bsdf_0']['alpha']['filename'] = maps_dir+'/rgh.png'
        self.scene['shape']['bsdf']['bsdf']['bsdf_0']['specular_reflectance']['filename'] = maps_dir+'/spe.png'
        self.scene['shape']['bsdf']['bsdf']['bsdf_1']['reflectance']['filename'] = maps_dir+'/dif.png'

    def scene_dict(self):
        scene = {
            'type': 'scene',
            'integrator': {
                'type': 'direct'
            },
            'sensor': {
                'type': 'perspective',
                'fov': 50,
                'to_world': T.look_at(
                    target=[0, 0, 0],
                    origin=[0, 0, 1],
                    up=[0, 1, 0]
                ),
                'sampler': {
                    'type': 'independent',
                    'sample_count': 256
                },
                'film': {
                    'type': 'hdrfilm',
                    'width': 256,
                    'height': 256,
                    'rfilter': {
                        'type': 'box'
                    },
                    'pixel_format': 'rgb'
                }
            },
            'shape': {
                'type': 'rectangle',
                'to_world': T.translate([0, 0, 0]).scale([0.3, 0.3, 0.3]),
                'bsdf': {
                    'type': 'normalmap',
                    'normalmap': {
                        'type': 'bitmap',
                        'filename': 'nom.png',
                        'to_uv': T.scale([1, -1, 1]),
                        'raw': True
                    },
                    'bsdf': {
                        'type': 'blendbsdf',
                        'weight': 0.5,
                        'bsdf_0': {
                            'type': 'roughconductor',
                            'distribution': 'ggx',
                            'alpha': {
                                'type': 'bitmap',
                                'filename': 'rgh.png',
                                'to_uv': T.scale([1, -1, 1]),
                                'raw': False
                            },
                            'specular_reflectance': {
                                'type': 'bitmap',
                                'filename': 'spe.png',
                                'to_uv': T.scale([1, -1, 1]),
                                'raw': False
                            }
                        },
                        'bsdf_1': {
                            'type': 'diffuse',
                            'reflectance': {
                                'type': 'bitmap',
                                'filename': 'dif.png',
                                'to_uv': T.scale([1, -1, 1]),
                                'raw': False
                            }
                        }
                    }
                }
            },
            'light': {
                    'type': 'point',
                    'position': [0, 0, 3],
                    'intensity': {
                        'type': 'rgb',
                        'value': 100
                    }
                },

        }

        return scene
