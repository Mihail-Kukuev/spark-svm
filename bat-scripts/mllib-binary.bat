spark-submit --class mllib.svm.MllibBinary ^
 --master local[4] ^
 D:\Google-drive\Thesis\program\spark-svm\mllib\build\svm-mllib.jar ^
 D:/Study/univer/thesis/libraries/datasets/rcv1_train.binary ^
 D:/Study/univer/thesis/libraries/datasets/rcv1_test.binary ^
 D:/tmp/results_mllib.txt ^
 1
pause