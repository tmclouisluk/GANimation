import argparse

import cv2
import os
import csv
import random

import face_recognition
from PIL import Image

import numpy as np
from utils import cv_utils, face_utils
from multiprocessing import Pool, Process, Manager
import threading
from queue import Queue

class CropFaces:
    def __init__(self, opt):
        self._opt = opt

    def morph_file(self, img_path):
        img = cv_utils.read_cv2_img(img_path)
        morphed_img = self._img_morph(img)
        output_name = os.path.basename(img_path)
        self._save_img(morphed_img, output_name)

    def _img_morph(self, img):
        bbs = face_recognition.face_locations(img)
        if len(bbs) > 0:
            y, right, bottom, x = bbs[0]
            bb = x, y, (right - x), (bottom - y)
            face = face_utils.crop_face_with_bb(img, bb)
            face = face_utils.resize_face(face)
        else:
            face = None
        return face

    def _save_img(self, img, filename):
        if img is not None:
            filepath = os.path.join(self._opt.output_dir, filename)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filepath, img)


def main():
    MAX_THREADS = 8
    print("Using %s threads" % MAX_THREADS)
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_folder', type=str, help='path to image')
    parser.add_argument('--output_dir', type=str, default='./output', help='output path')

    opt = parser.parse_args()

    if not os.path.isdir(opt.output_dir):
        os.makedirs(opt.output_dir)

    assert os.path.isdir(opt.images_folder), '%s is not a valid directory' % dir

    paths = []

    for root, _, fnames in sorted(os.walk(opt.images_folder)):
        for fname in fnames:
            path = os.path.join(root, fname)
            paths.append(path)

    paths_clusters = np.array_split(np.asarray(paths), MAX_THREADS)

    def crop_photo(p_c):
        for p in p_c:
            try:
                morph = CropFaces(opt)
                morph.morph_file(p)
                print(p)
            except Exception as err:
                pass

    manager = Manager()
    processes = []
    for c in range(MAX_THREADS):
        p = Process(target=crop_photo, args=(paths_clusters[c],))
        processes.append(p)
    # Start the processes
    for p in processes:
        p.start()
    # Ensure all processes have finished execution
    for p in processes:
        p.join()

    image_list = []
    for root, _, fnames in sorted(os.walk(opt.output_dir)):
        for fname in fnames:
            path = fname
            image_list.append(path)

    if len(image_list) > 0:
        train = random.sample(image_list, k=int(len(image_list) * 0.8))
        test = random.sample(image_list, k=int(len(image_list) * 0.2))
        if not os.path.isdir('%s/csv' % opt.output_dir):
            os.makedirs('%s/csv' % opt.output_dir)
        with open('%s/csv/train_ids.csv' % opt.output_dir, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for img in train:
                writer.writerow([img])
        with open('%s/csv/test_ids.csv' % opt.output_dir, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for img in test:
                writer.writerow([img])


if __name__ == '__main__':
    main()
