# Face Filter
```
Filter out faces you hate.
```
## Dependencies
- [torch](http://www.torch.ch)
- [cuDNN](https://developer.nvidia.com/cudnn)
- [turbo](https://luarocks.org/modules/kernelsauce/turbo)

## How to play
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
lua turbo http server will listen at port 12121(see source file web.lua)
```
$ cd face_filter_pipeline
$ th web.lua
```
## Training your own model
```
There are two models to be train. A face detector and a face classifier.
a. Training a face detector, enter 'face_detector_data_set' folder, put positive(face) and negative(not a face) example images in 'pos' and 'neg' folder, modify 'face_detector/data.lua' source file, line 6 'local videoLength = your pos examples count', then '$ cd face_detector; th run.lua', wait for training complete.
b. Training a face classifier, like training a detector, prepare the training example file: favorite_face_classifier_data_set/pos favorite_face_classifier_data_set/neg, modify favorite_face_classifier/data.lua, $ cd favorite_face_classifier; th run.lua.
c. Note, once you modify(add or remove) training example in pos or neg folder, you must delete 'training_data' in face_detector or favorite_face_classifier folder, let the program regenerate the training data.
```
