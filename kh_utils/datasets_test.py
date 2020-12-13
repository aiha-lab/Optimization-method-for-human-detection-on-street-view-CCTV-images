from .datasets import ImageFolderValidPipe, DALIValidIterator

# DATA_DIR = "../iitpdata/images/val/"
DATA_DIR = "../iitpdata_kh/images/"

pipeline = ImageFolderValidPipe(batch_size=128, num_threads=4, device_id=0, data_dir=DATA_DIR,
                                image_size=(360, 640), stride=32,
                                mean=0.0, stddev=255.0,  # [0, 255] -> [0, 1]
                                fp16=False, cpu_only=False)
pipeline.build()
loader = DALIValidIterator(pipeline)

print("Loader length", len(loader))
for i, (data, img_path) in enumerate(loader):
    print(i, type(data), data.shape, data.dtype, data.device)
    print(i, img_path[:4])
