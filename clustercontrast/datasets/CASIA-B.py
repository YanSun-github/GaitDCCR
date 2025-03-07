import glob
import os.path as osp
import re
import os
import json
from ..utils.data import BaseImageDataset


# def process_dir(dir_path, relabel=False):
#     img_paths = glob.glob(osp.join(dir_path, "*.jpg"))
#     pattern = re.compile(r"([-\d]+)_c(\d)")
#
#     # get all identities
#     pid_container = set()
#     for img_path in img_paths:
#         pid, _ = map(int, pattern.search(img_path).groups())
#         if pid == -1:
#             continue
#         pid_container.add(pid)
#
#     pid2label = {pid: label for label, pid in enumerate(pid_container)}
#
#     data = []
#     for img_path in img_paths:
#         pid, camid = map(int, pattern.search(img_path).groups())
#         if (pid not in pid_container) or (pid == -1):
#             continue
#
#         assert 1 <= camid <= 8
#         camid -= 1
#
#         if relabel:
#             pid = pid2label[pid]
#         data.append((img_path, pid, camid))
#
#     return data

def get_seqs_info_list(dir_path, label_set):
    seqs_info_list = []
    for lab in label_set:
        for typ in sorted(os.listdir(osp.join(dir_path, lab))):
            for vie in sorted(os.listdir(osp.join(dir_path, lab, typ))):
                seq_info = [lab, typ, vie]
                seq_path = osp.join(dir_path, *seq_info)
                seq_dirs = sorted(os.listdir(seq_path))
                if seq_dirs != []:
                    seq_dirs = [osp.join(seq_path, dir)
                                for dir in seq_dirs]
                    # if data_in_use is not None:
                    #     seq_dirs = [dir for dir, use_bl in zip(
                    #         seq_dirs, data_in_use) if use_bl]
                    seqs_info_list.append([*seq_info, seq_dirs])
    return seqs_info_list


def process_dir(dir_path, training):
    with open("../datasets/CASIA-B/CASIA-B.json", "rb") as f:
        partition = json.load(f)
    train_set = partition["TRAIN_SET"]
    test_set = partition["TEST_SET"]
    label_list = os.listdir(dir_path)
    train_set = [label for label in train_set if label in label_list]
    test_set = [label for label in test_set if label in label_list]
    data = get_seqs_info_list(dir_path, train_set) if training else get_seqs_info_list(dir_path, test_set)


    # img_paths = glob.glob(osp.join(dir_path, "*.jpg"))
    # pattern = re.compile(r"([-\d]+)_c(\d)")

    # get all identities
    # pid_container = set()
    # for img_path in img_paths:
    #     pid, _ = map(int, pattern.search(img_path).groups())
    #     if pid == -1:
    #         continue
    #     pid_container.add(pid)
    #
    # pid2label = {pid: label for label, pid in enumerate(pid_container)}

    # for img_path in img_paths:
    #     pid, camid = map(int, pattern.search(img_path).groups())
    #     if (pid not in pid_container) or (pid == -1):
    #         continue
    #
    #     assert 1 <= camid <= 8
    #     camid -= 1
    #
    #     if relabel:
    #         pid = pid2label[pid]
    #     data.append((img_path, pid, camid))

    return data

class CASIA_B(BaseImageDataset):

    """DukeMTMC-reID.
    Reference:
        - Ristani et al. Performance Measures and a Data Set for Multi-Target,
            Multi-Camera Tracking. ECCVW 2016.
        - Zheng et al. Unlabeled Samples Generated by GAN Improve the Person
            Re-identification Baseline in vitro. ICCV 2017.
    URL: `<https://github.com/layumi/DukeMTMC-reID_evaluation>`_

    Dataset statistics:
        - identities: 1404 (train + query).
        - images:16522 (train) + 2228 (query) + 17661 (gallery).
        - cameras: 8.
    """

    dataset_dir = "CASIA-B-pkl"

    def __init__(self, root, verbose=True):
        super(CASIA_B, self).__init__()
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        # self.query_dir = osp.join(self.dataset_dir, 'query')
        # self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        train = process_dir(self.dataset_dir, training=True)
        query = process_dir(self.dataset_dir, training=True)
        gallery = process_dir(self.dataset_dir, training=False)

        self.train = train
        self.query = query
        self.gallery = gallery

        # self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        # self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        # self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        # if not osp.exists(self.train_dir):
        #     raise RuntimeError("'{}' is not available".format(self.train_dir))
        # if not osp.exists(self.query_dir):
        #     raise RuntimeError("'{}' is not available".format(self.query_dir))
        # if not osp.exists(self.gallery_dir):
        #     raise RuntimeError("'{}' is not available".format(self.gallery_dir))
