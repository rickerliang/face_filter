cudnn.inception

==> doing epoch on training data:
==> epoch # 41 [batchSize = 420]
 [======================================== 82741/83060 ===============================>.]  ETA: 598ms | Step: 1ms
==> time to learn 1 sample = 1.8500618351358ms
ConfusionMatrix:
[[   41205     181]   99.563%   [class: pos]
 [     205   41149]]  99.504%   [class: neg]
 + average row correct: 99.533468484879%
 + average rowUcol correct (VOC measure): 99.071288108826%
 + global correct: 99.533478365966%
==> loss 3.5217542203736e-05
==> testing on test set:
 [======================================== 8821/9228 ==============================>....]  ETA: 0ms | Step: 23h59m
==> time to test 1 sample = 0.74645276187455ms
ConfusionMatrix:
[[    4325      85]   98.073%   [class: pos]
 [     161    4249]]  96.349%   [class: neg]
 + average row correct: 97.210884094238%
 + average rowUcol correct (VOC measure): 94.572746753693%
 + global correct: 97.210884353742%
==> loss 0.00035533512474781
==> best loss 0.00039796022758812 > 0.00035533512474781
==> save best model

total - 1000
total pos - 500
total neg - 500
true pos : 492
false pos : 14
true neg : 486
false neg : 8
sensitivity : true positive rate : recall : TPR : (true pos / total pos) :
.98
1-specificity :
.03
specificity : true negative rate : SPC : TNR : (true neg / total neg) :
.97
total - 1000
total pos - 500
total neg - 500
true pos : 494
false pos : 16
true neg : 484
false neg : 6
sensitivity : true positive rate : recall : TPR : (true pos / total pos) :
.98
1-specificity :
.04
specificity : true negative rate : SPC : TNR : (true neg / total neg) :
.96
total - 1000
total pos - 500
total neg - 500
true pos : 495
false pos : 21
true neg : 479
false neg : 5
sensitivity : true positive rate : recall : TPR : (true pos / total pos) :
.99
1-specificity :
.05
specificity : true negative rate : SPC : TNR : (true neg / total neg) :
.95
total - 1000
total pos - 500
total neg - 500
true pos : 495
false pos : 28
true neg : 472
false neg : 5
sensitivity : true positive rate : recall : TPR : (true pos / total pos) :
.99
1-specificity :
.06
specificity : true negative rate : SPC : TNR : (true neg / total neg) :
.94
total - 1000
total pos - 500
total neg - 500
true pos : 496
false pos : 30
true neg : 470
false neg : 4
sensitivity : true positive rate : recall : TPR : (true pos / total pos) :
.99
1-specificity :
.06
specificity : true negative rate : SPC : TNR : (true neg / total neg) :
.94

