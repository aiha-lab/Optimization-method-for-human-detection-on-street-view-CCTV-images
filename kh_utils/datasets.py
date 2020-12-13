import os
import math
import glob
from pathlib import Path
import numpy as np
import torch

from .general import xywh2xyxy, xyxy2xywh

from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIClassificationIterator
import nvidia.dali.ops as ops
import nvidia.dali.types as types

img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng']  # acceptable image suffixes


def create_dataloader(path, image_size, batch_size, stride, device_id=0, num_workers=8, queue_depth=8):
    pipeline = ImageFolderValidPipe(batch_size, num_workers, device_id, path, image_size, stride,
                                    mean=0.0, stddev=255.0, queue_depth=queue_depth)
    pipeline.build()
    loader = DALIValidIterator(pipeline)
    return loader


class ImageFolderValidPipe(Pipeline):
    """DALI GPU pipeline"""

    def __init__(self, batch_size, num_threads, device_id, data_dir, image_size, stride,
                 *, mean, stddev, shuffle=False, fp16=False, cpu_only=False, queue_depth=8):
        super(ImageFolderValidPipe, self).__init__(batch_size, num_threads, device_id, seed=-1)
        if isinstance(image_size, int):
            image_size = (image_size, image_size)  # (height, width)

        if (not isinstance(mean, float)) or (not isinstance(stddev, float)):
            assert len(mean) == len(stddev) == 3

        self.image_size = image_size

        try:
            f = []  # image files
            for p in data_dir if isinstance(data_dir, list) else [data_dir]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                elif p.is_file():  # file
                    with open(p, 'r') as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                else:
                    raise Exception('%s does not exist' % p)
            self.img_files = sorted([x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in img_formats])
            assert self.img_files, 'No images found'
        except Exception as e:
            raise Exception('Error loading data from {}: {}\n'.format(data_dir, e))

        img_files_stem = [Path(p).stem + ".jpg" for p in self.img_files]
        self.reader = ops.FileReader(file_root=data_dir, files=img_files_stem,
                                     prefetch_queue_depth=queue_depth,
                                     random_shuffle=shuffle)

        image_size = (int(math.ceil(image_size[0] / stride) * stride), int(math.ceil(image_size[1] / stride) * stride))

        if cpu_only:
            decode_device = 'cpu'
            self.dali_device = 'cpu'
        else:
            decode_device = 'mixed'
            self.dali_device = 'gpu'
        output_dtype = types.FLOAT16 if fp16 else types.FLOAT

        self.decode = ops.ImageDecoder(device=decode_device, output_type=types.RGB)
        self.resize = ops.Resize(device=self.dali_device, mode='default',
                                 size=image_size,
                                 interp_type=types.INTERP_LINEAR)
        # TODO this preserves the ratio.
        self.normalize = ops.Normalize(device=self.dali_device, dtype=output_dtype,
                                       mean=mean, stddev=stddev)

    def define_graph(self):
        self.jpegs, self.labels = self.reader(name="Reader")
        images = self.decode(self.jpegs)
        images = self.resize(images)
        output = self.normalize(images)
        self.labels = self.labels.gpu()
        return [output, self.labels]


class DALIValidIterator(object):

    def __init__(self, pipelines, num_examples=-1):
        self._dali_iterator = DALIClassificationIterator(pipelines=pipelines, size=num_examples,
                                                         last_batch_policy='PARTIAL',
                                                         reader_name="Reader")

        self.batch_size = pipelines.batch_size
        self.img_files = pipelines.img_files
        self.count = 0

    def __iter__(self):
        return self

    def __len__(self):
        return int(math.ceil(self._dali_iterator._size / self._dali_iterator.batch_size))

    def __next__(self):
        try:
            data = next(self._dali_iterator)
        except StopIteration:
            print("Resetting DALI loader")
            self._dali_iterator.reset()
            self.count = 0
            raise StopIteration

        input = data[0]['data']  # no label return, only image.
        _valid_count = min(self.count + self.batch_size, len(self.img_files)) - self.count
        img_files = self.img_files[self.count:self.count + _valid_count]
        self.count += _valid_count

        if input.shape[0] > _valid_count:
            input = input[:_valid_count]

        input = input.permute(0, 3, 1, 2)  # NHWC to NCHW

        return input, img_files
