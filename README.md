<img src='http://www.albertpumarola.com/images/2018/GANimation/face1_cyc.gif' align="right" width=90>

# GANimation: Anatomically-aware Facial Animation from a Single Image
### [[Project]](http://www.albertpumarola.com/research/GANimation/index.html)[ [Paper]](https://arxiv.org/abs/1807.09251) 
Official implementation of [GANimation](http://www.albertpumarola.com/research/GANimation/index.html). In this work we introduce a novel GAN conditioning scheme based on Action Units (AU) annotations, which describe in a continuous manifold the anatomical facial movements defining a human expression. Our approach permits controlling the magnitude of activation of each AU and combine several of them. For more information please refer to the [paper](https://arxiv.org/abs/1807.09251).

This code was made public to share our research for the benefit of the scientific community. Do NOT use it for immoral purposes.

![GANimation](http://www.albertpumarola.com/images/2018/GANimation/teaser.png)

## Prerequisites
- Install PyTorch (version 0.3.1), Torch Vision and dependencies from http://pytorch.org
- Install requirements.txt (```pip install -r requirements.txt```)

## Data Preparation
The code requires a directory containing the following files:
- `imgs/`: folder with all image
- `aus_openface.pkl`: dictionary containing the images action units.
- `train_ids.csv`: file containing the images names to be used to train.
- `test_ids.csv`: file containing the images names to be used to test.

An example of this directory is shown in `sample_dataset/`.

To crop imgs and generate `train_ids.csv` and `test_ids.csv` in `output_dir/csv`
run:
```
python pre_train --images_folder path_to_images --output_dir path_for_generated_files
```

To generate the `aus_openface.pkl` extract each image Action Units with [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Action-Units) and store each output in a csv file the same name as the image. Then run:
```
cd OpenFace 
cd build
./bin/FaceLandmarkImg -fdir path_to_images -out_dir path_for_generated_files -aus
```
```
python data/prepare_au_annotations.py -ia input_aus_filesdir -op output_path
```

## Run
To train:
```
python train.py --data_dir path/to/dataset/ --name experiment_1 --batch_size 25 
```
To test:
```
python test --input_path path/to/img --output_dir path/to/output --face_aus_path path/of/ausfile
```

## Citation
If you use this code or ideas from the paper for your research, please cite our paper:
```
@inproceedings{pumarola2018ganimation,
    title={GANimation: Anatomically-aware Facial Animation from a Single Image},
    author={A. Pumarola and A. Agudo and A.M. Martinez and A. Sanfeliu and F. Moreno-Noguer},
    booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
    year={2018}
}
```
