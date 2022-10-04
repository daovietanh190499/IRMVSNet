from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from pathlib import Path
import struct

def read_pfm(filename):
    with Path(filename).open('rb') as pfm_file:

        line1, line2, line3 = (pfm_file.readline().decode('latin-1').strip() for _ in range(3))
        assert line1 in ('PF', 'Pf')
        
        channels = 3 if "PF" in line1 else 1
        width, height = (int(s) for s in line2.split())
        scale_endianess = float(line3)
        bigendian = scale_endianess > 0
        scale = abs(scale_endianess)

        buffer = pfm_file.read()
        samples = width * height * channels
        assert len(buffer) == samples * 4
        
        fmt = f'{"<>"[bigendian]}{samples}f'
        decoded = struct.unpack(fmt, buffer)
        shape = (height, width, 3) if channels == 3 else (height, width)
        return np.flipud(np.reshape(decoded, shape)) * scale

# the DTU dataset preprocessed by Yao Yao (only for training)
class MVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, ndepths=192, interval_scale=1.06, **kwargs):
        super(MVSDataset, self).__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.ndepths = ndepths
        self.interval_scale = interval_scale

        assert self.mode in ["train", "val", "test"]
        self.metas = self.build_list()

    def build_list(self):
        metas = []
        with open(self.listfile) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

        # scans
        for scan in scans:
            pair_file = f"{scan}/cams/pair.txt"
            # read the pair file
            with open(os.path.join(self.datapath, pair_file)) as f:
                num_viewpoint = int(f.readline())
                # viewpoints (49)
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    metas.append((scan, ref_view, src_views))
        print("dataset", self.mode, "metas:", len(metas))
        return metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1]) * self.interval_scale
        return intrinsics, extrinsics, depth_min, depth_interval

    def read_img(self, filename):
        img = Image.open(filename)
        # scale 0~255 to 0~1
        np_img = np.array(img, dtype=np.float32) / 255.
        return np_img

    def read_depth(self, filename):
        # read pfm depth file
        return np.array(read_pfm(filename), dtype=np.float32)

    def __getitem__(self, idx):
        meta = self.metas[idx]
        scan, ref_view, src_views = meta
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.nviews - 1]

        imgs = []
        mask = None
        depth = None
        depth_values = None
        K = []
        Rt = []

        for i, vid in enumerate(view_ids):
            # NOTE that the id in image file names is from 1 to 49 (not 0~48)
            img_filename = os.path.join(self.datapath, '{}/blended_images/{}.jpg'.format(scan, str(vid).zfill(8)))
            mask_filename = os.path.join(self.datapath, '{}/blended_images/{}_masked.jpg'.format(scan, str(vid).zfill(8)))
            depth_filename = os.path.join(self.datapath, '{}/rendered_depth_maps/{}.pfm'.format(scan, str(vid).zfill(8)))
            proj_mat_filename = os.path.join(self.datapath, '{}/cams/{}_cam.txt').format(scan, str(vid).zfill(8))

            imgs.append(self.read_img(img_filename))
            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(proj_mat_filename)
            
            K.append(intrinsics)
            Rt.append(extrinsics)

            if i == 0:  # reference view
                depth_values = np.arange(depth_min, depth_interval * self.ndepths + depth_min, depth_interval,
                                         dtype=np.float32)
                mask = self.read_img(mask_filename)
                depth = self.read_depth(depth_filename)

        imgs = np.stack(imgs).transpose([0, 3, 1, 2])
        K = np.stack(K)
        Rt = np.stack(Rt)

        return {
            "id": scan,
            "imgs": imgs,
            "K": K,
            "Rt": Rt,
            "depth": depth,
            "depth_values": depth_values,
            "mask": mask
        }


if __name__ == "__main__":

    dataset = MVSDataset("../blended_mvs/", '../blended_mvs/train.txt', 'train', 10, 128)

    item = dataset[50]
    print(item.keys())
    print("id", item["id"])
    print("imgs", item["imgs"].shape)
    print("K", item["K"].shape)
    print("Rt", item["Rt"].shape)
    print("depth", np.min(item["depth"]))
    print("depth_values", item["depth_values"].shape)
    print("mask", item["mask"].shape)
