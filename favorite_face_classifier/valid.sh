#!/bin/bash

validationPath=$1

totalExample=$(ls -l validation_set/*.png | wc --line)
posExample=$(ls -l validation_set/like*.png | wc --line)
negExample=$(ls -l validation_set/unlike*.png | wc --line)

echo "total - $totalExample"
echo "total pos - $posExample"
echo "total neg - $negExample"

cfPosExample=$(ls -l $validationPath/like/*.png | wc --line)
cfNegExample=$(ls -l $validationPath/unlike/*.png | wc --line)

cfFalsePosExample=$(ls -l $validationPath/like/unlike*.png | wc --line)
cfFalseNegExample=$(ls -l $validationPath/unlike/like*.png | wc --line)

let cfTruePosExample=cfPosExample-cfFalsePosExample
let cfTrueNegExample=cfNegExample-cfFalseNegExample

echo "true pos : $cfTruePosExample"
echo "false pos : $cfFalsePosExample"
echo "true neg : $cfTrueNegExample"
echo "false neg : $cfFalseNegExample"

echo "sensitivity : true positive rate : recall : TPR : (true pos / total pos) : "
echo "scale=2; $cfTruePosExample / $posExample" | bc

echo "1-specificity : "
echo "scale=2; 1 - $cfTrueNegExample / $negExample" | bc

echo "specificity : true negative rate : SPC : TNR : (true neg / total neg) : "
echo "scale=2; $cfTrueNegExample / $negExample" | bc
