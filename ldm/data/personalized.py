import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import torch
from ldm.util import instantiate_from_config

import random

class PersonalizedBase(Dataset):
    def __init__(self,
                 data_root,
                 n=None,
                 cache=None,
                 cond_stage_config=None,
                 flip_p=0.5 # currently unused
                 ):

        self.data_root = data_root

        #self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(os.path.join(self.data_root, 'imgs'))]

        self._length = n
        
        
        """
        # done in preprocessing already
        self.size = size 
        self.interpolation = {# "linear": PIL.Image.Resampling.LINEAR, # Linear no longer exists? 
                              "bilinear": PIL.Image.Resampling.BILINEAR,
                              "bicubic": PIL.Image.Resampling.BICUBIC,
                              "lanczos": PIL.Image.Resampling.LANCZOS,
                              }[interpolation]
        """
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)
        self.cache = cache

        self.cond_stage_config = cond_stage_config
        self.initData()

    def initData(self):
        if self.cache:
            cachepath = os.path.join(self.data_root, self.cache)
            if os.path.exists(cachepath):
                print("Loading dataset from cache...")
                self._data = torch.load(cachepath)
                if self._length is not None:
                    random.shuffle(self._data)
                    self._data = self._data[:min(len(self._data), self._length)]
                self._length = len(self._data)
                return

        if self.cond_stage_config:
            model = instantiate_from_config(self.cond_stage_config).cuda()

        df = pd.read_parquet(os.path.join(self.data_root, 'laion5obj.parquet'), engine='fastparquet') 
        df = df.sample(frac=1).reset_index(drop=True) # shuffle dataset
        if self._length is not None:
            df = df.head(self._length)

        def initImage(key, caption):
            example = {}
            # columns: index similarity hash punsafe pwatermark aesthetic caption url key status error_message width height original_width original_height exif sha256
            image = Image.open(os.path.join(self.data_root, 'imgs', key+'.jpg'))
            if not image.mode == "RGB":
                image = image.convert("RGB")
            #img = np.array(image).astype(np.uint8)
            #image = Image.fromarray(img)
            #image = self.flip(image)
            image = np.array(image).astype(np.uint8)
            #image = (image / 255).astype(np.float32) # TODO: or 16? and remove offset
            image = (image / 127.5 - 1.0).astype(np.float32) # TODO: or 16? and remove offset
            example["image"] =  torch.from_numpy(np.concatenate((image, 1-image), axis=2))
            
            if self.cond_stage_config:
                example["caption"] = model.encode(caption)[0].cpu()
            else:
                example["caption"] = caption 
            return example

        self._data = [initImage(x, y) for x, y in zip(df['key'], df['caption'])]
        del model
        
        if self.cache:
            cachepath = os.path.join(self.data_root, self.cache)
            torch.save(self._data, cachepath)
        
        self._length = len(self._data)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        sample = self._data[i % self._length]
        #sample["image"] = self.flip(sample["image"]) # it seems to work but based on documentation it shouldn't?
        return sample