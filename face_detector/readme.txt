normalize yes/no
training set : no
classifier : no
model : no
validation set : no

model 
kernel reduce 
convlayer 4
kernel width 4
pooling 4 step 3

feature dim
256

classifier
256
2

do not use m4v stream, use image file directly
save train and test falsePos falseNeg

==> doing epoch on training data:
==> epoch # 36 [batchSize = 128]
 [============================== 61046/61046 =====================>]  Tot: 31s272ms | Step: 0ms

==> time to learn 1 sample = 0.5124847506905ms
ConfusionMatrix:
[[   30422      36]   99.882%   [class: pos]
 [      27   30443]]  99.911%   [class: neg]
 + average row correct: 99.896594882011%
 + average rowUcol correct (VOC measure): 99.793410301208%
 + global correct: 99.896599264706%
==> loss 2.7412548898159e-05
==> testing on test set:
 [============================== 6782/6782 =======================>]  Tot: 1s802ms | Step: 0ms

==> time to test 1 sample = 0.26685194574039ms
ConfusionMatrix:
[[    3312      16]   99.519%   [class: pos]
 [       7    3321]]  99.790%   [class: neg]
 + average row correct: 99.654445052147%
 + average rowUcol correct (VOC measure): 99.311271309853%
 + global correct: 99.654447115385%
==> loss 7.2711487449012e-05
==> previous loss 7.9200322103973e-05 >  current loss 7.2711487449012e-05
==> save best model

