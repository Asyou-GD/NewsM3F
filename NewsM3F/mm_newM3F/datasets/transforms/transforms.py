import warnings
from typing import Optional
import mmcv
import numpy as np
from mmcv import LoadImageFromFile
from mmpretrain.registry import TRANSFORMS
from tools_redecology.utils.redkv_cache_utils import get_urls_contents


@TRANSFORMS.register_module()
class LoadImageFromUrl(LoadImageFromFile):
    def __init__(self, mean_rgb: Optional[list] = None, load_size=1080, overwrite=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if mean_rgb is None:
            mean_rgb = [0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255]
        mean_bgr = mean_rgb[::-1]
        self.mean_bgr = np.array([[mean_bgr]], dtype=np.float32)
        self.load_size = load_size
        self.overwrite = overwrite

    def transform(self, results: dict) -> Optional[dict]:
        filename = results['url']

        # empty url
        if len(filename) < 3:
            img = np.ones((224, 224, 3)) * self.mean_bgr
        else:
            try:
                img = self.load_img_from_redkv(filename, is_log=False)
            except Exception as e:
                if self.ignore_empty:
                    return None
                else:
                    raise e
        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        results['img_path'] = filename
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'ignore_empty={self.ignore_empty}, '
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f"imdecode_backend='{self.imdecode_backend}', ")

        if self.file_client_args is not None:
            repr_str += f'file_client_args={self.file_client_args})'
        else:
            repr_str += f'backend_args={self.backend_args})'

        return repr_str

    def load_img_from_redkv(self, url, max_retry=1, is_log=False):
        for i in range(max_retry):
            try:
                img_bytes = get_urls_contents(url, parse_key_fn=None, group='ut', expire_time=720*3, overwrite=self.overwrite,
                                              log=is_log, img_size=self.load_size)
                img = mmcv.imfrombytes(img_bytes[0], flag=self.color_type, backend=self.imdecode_backend)
                break
            except Exception as e:
                img = None
                # print('download {} retry {}'.format(url, i), flush=True)
                
        if img is None:
            raise Exception('Error! Download {} fail'.format(url))
        return img