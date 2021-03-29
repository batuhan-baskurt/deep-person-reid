from __future__ import division, print_function, absolute_import
import re
import glob
import os.path as osp
import warnings

from ..dataset import ImageDataset


class APM(ImageDataset):
    """APM.

    Reference:
        Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.

    URL: `<http://www.liangzheng.org/Project/project_reid.html>`_
    
    Dataset statistics:
        - identities: 1501 (+1 for background).
        - images: 12936 (train) + 3368 (query) + 15913 (gallery).
    """
    _junk_pids = [0, -1]
    dataset_dir = 'APM'
    #dataset_url = 'http://188.138.127.15:81/Datasets/Market-1501-v15.09.15.zip'

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        #self.download_dataset(self.dataset_dir, self.dataset_url)

        # allow alternative directory structure
        self.data_dir = self.dataset_dir

        self.train_dir = osp.join(self.data_dir, 'bbox_train')
        self.query_dir = osp.join(self.data_dir, 'bbox_query')
        self.gallery_dir = osp.join(self.data_dir, 'bbox_test')

        required_files = [
            self.data_dir, self.train_dir, self.query_dir, self.gallery_dir
        ]

        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir, relabel=True)
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False)

        super(APM, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            
            elems = img_path.split('/')
            if elems:
                file_name = elems[-1]          
                camid = int(file_name[6:9])
            #print(camid)
            #print(pid)
            #assert 1 <= camid <= 8
            #camid -= 1 # index starts from 0
            if relabel:
                pid = pid2label[pid]
            #print(camid)
            #print(pid)
            data.append((img_path, pid, camid))

        return data

'''    def process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([\d]+)_c(\d\d\d)')

        data = []
        for img_path in img_paths:
            #pid, camid = map(int, pattern.search(img_path).groups())
            elems = img_path.split('/')
            if not elems:
                continue
            file_name = elems[-1]
            pid = int(file_name[0:4])
            # TODO: does not work for cam ids different than 3 digits
            camid = int(file_name[6:9])
            #print(file_name)
            #print(camid)
            #print(pid)
            if pid == -1: continue  # junk images are just ignored
            #assert 0 <= pid <= 776
            assert 1 <= camid <= 20
            camid -= 1  # index starts from 0
            #if is_train:
            #pid = self.dataset_name + "_" + str(pid)
            #camid = self.dataset_name + "_" + str(camid)
            data.append((img_path, pid, camid))

        return data'''
