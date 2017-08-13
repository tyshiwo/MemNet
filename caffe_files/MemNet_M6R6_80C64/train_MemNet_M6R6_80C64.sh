#!/usr/bin/env sh
LOG=./log/MemNet_M6R6_80C64_291_31.log
CAFFE=/data2/taiying/MSU_Code/120-caffe-master/build/tools/caffe # your caffe path

$CAFFE train --solver=./MemNet_M6R6_80C64_solver.prototxt -gpu 0 2>&1 | tee $LOG
