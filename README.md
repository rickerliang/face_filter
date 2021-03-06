# Face Filter
```
Filter out faces you hate.
```
## Dependencies
- [torch](http://www.torch.ch)
- [cuDNN](https://developer.nvidia.com/cudnn)
- [turbo](https://luarocks.org/modules/kernelsauce/turbo)
- [graphicsmagick](https://github.com/torch/rocks/blob/master/graphicsmagick-1.scm-0.rockspec)
- [uuid](https://luarocks.org/modules/tieske/uuid)
- [dpnn](https://github.com/Element-Research/dpnn)
```bash
# in a terminal, run the commands WITHOUT sudo
PREFIX=$HOME/torch/install luarocks install turbo
luarocks install graphicsmagick
luarocks install uuid
luarocks install dpnn
```

## How to play
* Download pretrained classifier model from http://pan.baidu.com/s/1sl6MQLn.
```
$ cd face_filter_pipeline
$ th run.lua folder_to_classify
or
$ th test.lua pic_to_classify.jpg

to help
$ cd face_filter_pipeline
$ th run.lua --help
or
$ th test.lua --help
```
## Play with http
* lua turbo http server will listen at port 12121(see source file web.lua).
http://localhost:12121/index.html.
```
$ cd face_filter_pipeline
$ th web.lua
```
## Some Examples
* Images from http://image.baidu.com.
* Good faces will be surrounded by green bounding box, blue bounding box means poor face.
![slide](https://raw.githubusercontent.com/rickerliang/face_filter/master/readme_images/negative_1.png)
![slide](https://raw.githubusercontent.com/rickerliang/face_filter/master/readme_images/negative_2.png)
![slide](https://raw.githubusercontent.com/rickerliang/face_filter/master/readme_images/positive_1.png)
![slide](https://raw.githubusercontent.com/rickerliang/face_filter/master/readme_images/positive_2.png)

## Training your own model
```
There are two models to be trained. A face detector and a face classifier.
a. Training a face detector, enter 'face_detector_data_set' folder, put positive(face) and negative(not a face) example images in 'pos' and 'neg' folder, modify 'face_detector/data.lua' source file, line 6 'local videoLength = your pos examples count', then '$ cd face_detector; th run.lua', wait for training complete.
b. Training a face classifier, like training a detector, prepare the training example file: favorite_face_classifier_data_set/pos favorite_face_classifier_data_set/neg, modify favorite_face_classifier/data.lua, $ cd favorite_face_classifier; th run.lua.
c. Note, once you modify(add or remove) training example in pos or neg folder, you must delete 'training_data' in face_detector or favorite_face_classifier folder, let the program regenerate the training data.
```
## About model ensemble
* If you trained some models with different architecture or different hyper parameter, you can ensembling these models to achieve better performance than a single model.
* See [KAGGLE ENSEMBLING GUIDE](http://mlwave.com/kaggle-ensembling-guide/) for more detial.
* The face classifier supports model ensemble, put some trained models in `favorite_face_classifier/ensemble` folder, and `th classifier.lua -ensemble` to use model ensemble.
* In my experiment, 5 models ensemble achieve true positive rate 0.99 and true negative rate 0.98, in contrast, on a single model, true positive rate 0.98 and true negative rate 0.97 .
