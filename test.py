import os
import argparse
import glob
import random

import cv2
from utils import face_utils
from utils import cv_utils
import face_recognition
from PIL import Image
import torchvision.transforms as transforms
import torch
import pickle
import numpy as np
from models.models import ModelsFactory
from options.test_options import TestOptions

class MorphFacesInTheWild:
    def __init__(self, opt):
        self._opt = opt
        self._model = ModelsFactory.get_by_name(self._opt.model, self._opt)
        self._model.set_eval()
        self._transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                   std=[0.5, 0.5, 0.5])
                                              ])

    def morph_file(self, img_path, expresion, i, j=0):
        img = cv_utils.read_cv2_img(img_path)
        morphed_img, bbs = self._img_morph(img, expresion)
        overlay_fake_img = self._overlay_fake_img(img, morphed_img['fake_imgs_masked'], bbs)
        output_name = '%s_%s_out.png' % (os.path.basename(img_path), i)
        fake_output_name = '%s_out%s%s.png' % (os.path.basename(img_path), i, j)
        self._save_img(overlay_fake_img, fake_output_name)
        return fake_output_name
        #self._save_img(morphed_img['concat'], output_name)

    def _img_morph(self, img, expresion):
        bbs = face_recognition.face_locations(img)
        if len(bbs) > 0:
            y, right, bottom, x = bbs[0]
            bb = x, y, (right - x), (bottom - y)
            face = face_utils.crop_face_with_bb(img, bb)
            face = face_utils.resize_face(face)
        else:
            face = face_utils.resize_face(img)

        morphed_face = self._morph_face(face, expresion)

        return morphed_face, bbs

    def _morph_face(self, face, expresion):
        face = torch.unsqueeze(self._transform(Image.fromarray(face)), 0)
        expresion = torch.unsqueeze(torch.from_numpy(expresion/5.0), 0)
        test_batch = {'real_img': face, 'real_cond': expresion, 'desired_cond': expresion, 'sample_id': torch.FloatTensor(), 'real_img_path': []}
        self._model.set_input(test_batch)
        imgs, _ = self._model.forward(keep_data_for_visuals=False, return_estimates=True)
        return imgs

    def _overlay_fake_img(self, img, face, bbs):
        overlay_copy = img
        if len(bbs) > 0:
            y, right, bottom, x = bbs[0]
            ratio = ((bottom - y) / face.shape[0], (right - x) / face.shape[0])
            face = face_utils.resize_face(face, (int(face.shape[0]*ratio[0]), int(face.shape[1]*ratio[1])))
            #print(y, right, bottom, x)
            #print(overlay_copy.shape)
            overlay_copy[y:y + face.shape[0], x:x +face.shape[1]] = face
        return overlay_copy

    def _save_img(self, img, filename):
        filepath = os.path.join(self._opt.output_dir, filename)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filepath, img)


def main():
    opt = TestOptions().parse()
    if not os.path.isdir(opt.output_dir):
        os.makedirs(opt.output_dir)

    morph = MorphFacesInTheWild(opt)

    image_path = opt.input_path
    face_aus_path = opt.face_aus_path

    with open(face_aus_path, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        p = u.load()

    for i in range(0, opt.cond_nc):
        # expression = np.random.uniform(0, 1, opt.cond_nc)
        expressions = p[i]
        expression = random.choice(list(expressions.values()))

        diff = [x/10 for x in expression]
        print(expression)
        expression_images = []
        for j in range(10):
            expression_j = np.asarray([x*(j+1.0) for x in diff])
            print(expression_j)
            expression_images.append(morph.morph_file(image_path, expression_j, i, j))

        #im1 = Image.open(image_path)
        expression_images_reversed = list(reversed(expression_images))
        expression_images.extend(expression_images_reversed)
        
        frames = [Image.open(os.path.join(opt.output_dir, x)) for x in expression_images]
        gif_filepath = os.path.join(opt.output_dir, '%s_out%s.gif' % (os.path.basename(image_path), i))
        frames[0].save(gif_filepath, format='GIF', save_all=True, append_images=frames[1:], duration=100, loop=0)





if __name__ == '__main__':
    main()
