import os
# os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import torch
import trimesh
import numpy as np
import neural_renderer as nr
from torchvision.utils import make_grid
from utils.densepose_methods import DensePoseMethods
from skimage.transform import resize

# try:
#     import pyrender
# except:
#     pass
try:
    from opendr.renderer import ColoredRenderer
    from opendr.lighting import LambertianPointLight, SphericalHarmonics
    from opendr.camera import ProjectPoints
except:
    pass

try:
    pass
    # import taichi as ti
    # import taichi_three as t3
    # ti.init(ti.cuda)
except:
    pass

import logging
logger = logging.getLogger(__name__)

class PyRenderer:
    """
    Renderer used for visualizing the SMPL model
    Code adapted from https://github.com/vchoutas/smplify-x
    """
    def __init__(self, focal_length=5000, img_res=224, faces=None):
        self.renderer = pyrender.OffscreenRenderer(viewport_width=img_res,
                                       viewport_height=img_res,
                                       point_size=1.0)
        self.focal_length = focal_length
        self.camera_center = [img_res // 2, img_res // 2]
        self.faces = faces

    def visualize_tb(self, vertices, camera_translation, images):
        vertices = vertices.cpu().numpy()
        camera_translation = camera_translation.cpu().numpy().copy()
        images = images.cpu()
        images_np = np.transpose(images.numpy(), (0,2,3,1))
        rend_imgs = []
        for i in range(vertices.shape[0]):
            rend_img = torch.from_numpy(np.transpose(self.__call__(vertices[i], camera_translation[i], images_np[i]), (2,0,1))).float()
            rend_imgs.append(images[i])
            rend_imgs.append(rend_img)
        rend_imgs = make_grid(rend_imgs, nrow=2)
        return rend_imgs

    def __call__(self, vertices, camera_translation, image):
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            alphaMode='OPAQUE',
            baseColorFactor=(0.8, 0.3, 0.3, 1.0))

        camera_translation[0] *= -1.

        mesh = trimesh.Trimesh(vertices, self.faces)
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(ambient_light=(0.5, 0.5, 0.5))
        scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation
        camera = pyrender.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
                                           cx=self.camera_center[0], cy=self.camera_center[1])
        scene.add(camera, pose=camera_pose)

        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
        light_pose = np.eye(4)

        light_pose[:3, 3] = np.array([0, -1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([0, 1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([1, 1, 2])
        scene.add(light, pose=light_pose)

        color, rend_depth = self.renderer(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        valid_mask = (rend_depth > 0)[:,:,None]
        output_img = (color[:, :, :3] * valid_mask +
                  (1 - valid_mask) * image)
        return output_img


class OpenDRenderer:
    def __init__(self, resolution=(224, 224), ratio=1):
        self.resolution = (resolution[0] * ratio, resolution[1] * ratio)
        self.ratio = ratio
        self.focal_length = 5000.
        self.K = np.array([[self.focal_length, 0., self.resolution[1] / 2.],
                          [0., self.focal_length, self.resolution[0] / 2.],
                          [0., 0., 1.]])
        self.colors_dict = {
            'red': np.array([0.5, 0.2, 0.2]),
            'pink': np.array([0.7, 0.5, 0.5]),
            'neutral': np.array([0.7, 0.7, 0.6]),
            'purple': np.array([0.5, 0.5, 0.7]),
            'green': np.array([0.5, 0.55, 0.3]),
            'sky': np.array([0.3, 0.5, 0.55]),
            'white': np.array([1.0, 0.98, 0.94]),
        }
        self.renderer = ColoredRenderer()
    
    def reset_res(self, resolution):
        self.resolution = (resolution[0] * self.ratio, resolution[1] * self.ratio)
        self.K = np.array([[self.focal_length, 0., self.resolution[1] / 2.],
                          [0., self.focal_length, self.resolution[0] / 2.],
                          [0., 0., 1.]])

    def __call__(self, verts, faces, color=None, color_type='white', R=None, mesh_filename=None,
                image=np.zeros((224, 224, 3)), cam=np.array([1, 0, 0]),
                rgba=False, addlight=True):
        '''Render mesh using OpenDR
        verts: shape - (V, 3)
        faces: shape - (F, 3)
        image: shape - (224, 224, 3), range - [0, 255] (np.uint8)
        axis: rotate along with X/Y/Z axis (by angle)
        R: rotation matrix (used to manipulate verts) shape - [3, 3]
        Return:
            rendered image: shape - (224, 224, 3), range - [0, 255] (np.uint8)
        '''
        ## Create OpenDR renderer
        rn = self.renderer
        h, w = self.resolution
        K = self.K

        f = np.array([K[0, 0], K[1, 1]])
        c = np.array([K[0, 2], K[1, 2]])
        if len(cam) == 4:
            t = np.array([cam[2], cam[3], 2 * K[0, 0] / (w * cam[0] + 1e-9)])
        elif len(cam) == 3:
            t = np.array([cam[1], cam[2], 2 * K[0, 0] / (w * cam[0] + 1e-9)])
    
        rn.camera = ProjectPoints(rt=np.array([0, 0, 0]), t=t, f=f, c=c, k=np.zeros(5))
        rn.frustum = {'near': 1., 'far': 1000., 'width': w, 'height': h}

        albedo = np.ones_like(verts)*.9

        # if color is not None:
        #     if len(color.shape) == 1:
        #         color = np.ones(verts.shape) * color[None, :]
        # else:
        #     color = np.ones_like(verts) * self.colors_dict[color_type][None, :]

        if color is not None:
            color0 = np.array(color)
            color1 = np.array(color)
            color2 = np.array(color)
        elif color_type == 'white':
            color0 = np.array([1., 1., 1.])
            color1 = np.array([1., 1., 1.])
            color2 = np.array([0.7, 0.7, 0.7])
            color = np.ones_like(verts) * self.colors_dict[color_type][None, :]
        else:
            color0 = self.colors_dict[color_type] * 1.2
            color1 = self.colors_dict[color_type] * 1.2
            color2 = self.colors_dict[color_type] * 1.2
            color = np.ones_like(verts) * self.colors_dict[color_type][None, :]

        # render_smpl = rn.r
        if R is not None:
            assert R.shape == (3, 3), "Shape of rotation matrix should be (3, 3)"
            verts = np.dot(verts, R)

        rn.set(v=verts, f=faces, vc=color, bgcolor=np.zeros(3))

        if addlight:
            yrot = np.radians(120) # angle of lights
            # # 1. 1. 0.7
            rn.vc = LambertianPointLight(
                f=rn.f,
                v=rn.v,
                num_verts=len(rn.v),
                light_pos=rotateY(np.array([-200, -100, -100]), yrot),
                vc=albedo,
                light_color=color0)

            # Construct Left Light
            rn.vc += LambertianPointLight(
                f=rn.f,
                v=rn.v,
                num_verts=len(rn.v),
                light_pos=rotateY(np.array([800, 10, 300]), yrot),
                vc=albedo,
                light_color=color1)

            # Construct Right Light
            rn.vc += LambertianPointLight(
                f=rn.f,
                v=rn.v,
                num_verts=len(rn.v),
                light_pos=rotateY(np.array([-500, 500, 1000]), yrot),
                vc=albedo,
                light_color=color2)

        rendered_image = rn.r
        visibility_image = rn.visibility_image

        if type(image) is not list:
            image_list = [image]
        else:
            image_list = image
        
        return_img = []
        for img in image_list:
            if self.ratio != 1:
                img_resized = resize(img, (img.shape[0] * self.ratio, img.shape[1] * self.ratio), anti_aliasing=True)
            else:
                img_resized = img / 255.

            try:
                img_resized[visibility_image != (2**32 - 1)] = rendered_image[visibility_image != (2**32 - 1)]
            except:
                logger.warning('Can not render mesh.')

            img_resized = (img_resized * 255).astype(np.uint8)
            res = img_resized

            if rgba:
                img_resized_rgba = np.zeros((img_resized.shape[0], img_resized.shape[1], 4))
                img_resized_rgba[:, :, :3] = img_resized
                img_resized_rgba[:, :, 3][visibility_image != (2**32 - 1)] = 255
                res = img_resized_rgba.astype(np.uint8)
            return_img.append(res)

        if type(image) is not list:
            return_img = return_img[0]

        return return_img

class TaichiRenderer:
    def __init__(self, faces, v_n, resolution=(224, 224), out_h_w=(224, 224), ratio=1):
        self.resolution = (resolution[1] * ratio, resolution[0] * ratio)
        self.out_h_w = out_h_w
        self.ratio = ratio
        self.focal_length = 5000.
        self.K = np.array([[self.focal_length, 0., self.resolution[0] / 2.],
                          [0., self.focal_length, self.resolution[1] / 2.],
                          [0., 0., 1.]])
        self.colors_dict = {
            'red': np.array([0.5, 0.2, 0.2]),
            'pink': np.array([0.7, 0.5, 0.5]),
            'neutral': np.array([0.7, 0.7, 0.6]),
            'purple': np.array([0.5, 0.5, 0.7]),
            'green': np.array([0.5, 0.55, 0.3]),
            'sky': np.array([0.3, 0.5, 0.55]),
            'white': np.array([1.0, 0.98, 0.94]),
        }

        self.scene = t3.Scene()
        self.camera = t3.Camera(res=self.resolution)
        # model = t3.Model(obj=t3.readobj('test.obj'))
        f_n = len(faces)
        self.model = t3.Model(f_n=f_n, vi_n=v_n)

        self.light = t3.Light()
        self.scene.add_light(self.light)
        self.scene.add_model(self.model)
        self.scene.add_camera(self.camera)
        self.scene.init()

        self.model.faces.from_numpy(faces)

    def reset_res(self, resolution):
        self.resolution = (resolution[1] * self.ratio, resolution[0] * self.ratio)
        self.K = np.array([[self.focal_length, 0., self.resolution[0] / 2.],
                          [0., self.focal_length, self.resolution[1] / 2.],
                          [0., 0., 1.]])

    def __call__(self, verts, faces, color=None, color_type='white', R=None, mesh_filename=None,
                image=np.zeros((224, 224, 3)), cam=np.array([1, 0, 0]),
                rgba=False, addlight=True):
        '''Render mesh using OpenDR
        verts: shape - (V, 3)
        faces: shape - (F, 3)
        image: shape - (224, 224, 3), range - [0, 255] (np.uint8)
        axis: rotate along with X/Y/Z axis (by angle)
        R: rotation matrix (used to manipulate verts) shape - [3, 3]
        Return:
            rendered image: shape - (224, 224, 3), range - [0, 255] (np.uint8)
        '''
        ## Create OpenDR renderer
        w, h = self.resolution
        K = self.K
        verts = verts[0]

        f = np.array([K[0, 0], K[1, 1]])
        c = np.array([K[0, 2], K[1, 2]])
        if len(cam) == 4:
            t = np.array([cam[2], cam[3], 2 * K[0, 0] / (w * cam[0] + 1e-9)])
        elif len(cam) == 3:
            t = np.array([cam[1], cam[2], 2 * K[0, 0] / (w * cam[0] + 1e-9)])

        self.model.vi.from_numpy(verts)

        extrinsic = np.zeros((3, 4), dtype=float)
        rot = np.eye(3, dtype=float)
        t = -rot @ t
        extrinsic[:3, :3] = rot
        extrinsic[:3, 3] = t

        self.camera.set_intrinsic(K[0, 0]*self.ratio, K[0, 0]*self.ratio, w // 2, h // 2)
        self.camera._init()
        self.camera.pos.from_numpy(t)
        self.camera.trans.from_numpy(rot)

        # render the model(s) into image
        self.scene.render()

        # ti.imwrite(self.camera.img, 'test.jpg')
        rendered_img = self.camera.img.to_numpy().swapaxes(0, 1)

        rendered_img = (rendered_img * 255).astype(np.uint8)
        rendered_img = rendered_img[self.out_h_w[1] // 2 - self.out_h_w[0] // 2:self.out_h_w[1] // 2 + self.out_h_w[0] // 2]

        if type(image) is not list:
            image_list = [image]
        else:
            image_list = image
        
        return_img = []
        for img in image_list:
            img[rendered_img != 0] = rendered_img[rendered_img != 0]
            return_img.append(img)

        if type(image) is not list:
            return_img = return_img[0]

        return return_img

#  https://github.com/classner/up/blob/master/up_tools/camera.py
def rotateY(points, angle):
    """Rotate all points in a 2D array around the y axis."""
    ry = np.array([
        [np.cos(angle),     0.,     np.sin(angle)],
        [0.,                1.,     0.           ],
        [-np.sin(angle),    0.,     np.cos(angle)]
    ])
    return np.dot(points, ry)

def rotateX( points, angle ):
    """Rotate all points in a 2D array around the x axis."""
    rx = np.array([
        [1.,    0.,                 0.           ],
        [0.,    np.cos(angle),     -np.sin(angle)],
        [0.,    np.sin(angle),     np.cos(angle) ]
    ])
    return np.dot(points, rx)

def rotateZ( points, angle ):
    """Rotate all points in a 2D array around the z axis."""
    rz = np.array([
        [np.cos(angle),     -np.sin(angle),     0. ],
        [np.sin(angle),     np.cos(angle),      0. ],
        [0.,                0.,                 1. ]
    ])
    return np.dot(points, rz)


class IUV_Renderer(object):
    def __init__(self, focal_length=5000., orig_size=224, output_size=56):

        self.focal_length = focal_length
        self.orig_size = orig_size

        DP = DensePoseMethods()

        vert_mapping = DP.All_vertices.astype('int64') - 1
        self.vert_mapping = torch.from_numpy(vert_mapping)

        faces = DP.FacesDensePose
        faces = faces[None, :, :]
        self.faces = torch.from_numpy(faces.astype(np.int32))

        num_part = float(np.max(DP.FaceIndices))
        textures = np.array(
            [(DP.FaceIndices[i] / num_part, np.mean(DP.U_norm[v]), np.mean(DP.V_norm[v])) for i, v in
             enumerate(DP.FacesDensePose)])

        textures = textures[None, :, None, None, None, :]
        self.textures = torch.from_numpy(textures.astype(np.float32))

        self.renderer = nr.Renderer(camera_mode='projection', image_size=output_size, fill_back=False, anti_aliasing=False,
                                    dist_coeffs=torch.FloatTensor([[0.] * 5]), orig_size=self.orig_size)
        self.renderer.light_intensity_directional = 0.0
        self.renderer.light_intensity_ambient = 1.0

        K = np.array([[self.focal_length, 0., self.orig_size / 2.],
                      [0., self.focal_length, self.orig_size / 2.],
                      [0., 0., 1.]])

        R = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])

        t = np.array([0, 0, 5])

        if self.orig_size != 224:
            rander_scale = self.orig_size / float(224)
            K[0, 0] *= rander_scale
            K[1, 1] *= rander_scale
            K[0, 2] *= rander_scale
            K[1, 2] *= rander_scale

        self.K = torch.FloatTensor(K[None, :, :])
        self.R = torch.FloatTensor(R[None, :, :])
        self.t = torch.FloatTensor(t[None, None, :])


    def camera_matrix(self, cam):
        batch_size = cam.size(0)

        K = self.K.repeat(batch_size, 1, 1)
        R = self.R.repeat(batch_size, 1, 1)
        # t = self.t.repeat(batch_size, 1, 1)
        t = torch.stack([cam[:, 1], cam[:, 2], 2 * self.focal_length/(self.orig_size * cam[:, 0] + 1e-9)], dim=-1)
        t = t.unsqueeze(1)

        if cam.is_cuda:
            device_id = cam.get_device()
            K = K.cuda(device_id)
            R = R.cuda(device_id)
            t = t.cuda(device_id)

        return K, R, t

    def verts2iuvimg(self, verts, cam):
        batch_size = verts.size(0)

        K, R, t = self.camera_matrix(cam)

        if self.vert_mapping is None:
            vertices = verts
        else:
            vertices = verts[:, self.vert_mapping, :]

        iuv_image = self.renderer(vertices, self.faces.to(verts.device).expand(batch_size, -1, -1),
                               self.textures.to(verts.device).expand(batch_size, -1, -1, -1, -1, -1).clone(),
                               K=K, R=R, t=t,
                               mode='rgb',
                               dist_coeffs=torch.FloatTensor([[0.] * 5]).to(verts.device))

        return iuv_image