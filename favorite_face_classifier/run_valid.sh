rm -rf classify_output*
th classify.lua -input validation_set -threshold 0.8;th classify.lua -input validation_set -threshold 0.6;th classify.lua -input validation_set -threshold 0.4;th classify.lua -input validation_set -threshold 0.2;./valid.sh classify_output0.8;./valid.sh classify_output0.6;./valid.sh classify_output0.4;./valid.sh classify_output0.2;
