spark-submit --class liblinear.svm.LiblinearBinary ^
 --master local[4] ^
 --jars D:\Google-drive\Thesis\program\spark-svm\liblinear\build\svm-liblinear.jar,D:\Google-drive\Thesis\program\spark-svm\mllib\build\svm-mllib.jar, D:\tmp\spark-liblinear-1.96.jar ^
 D:\Study\univer\thesis\libraries\datasets\rcv1_train.binary ^
 D:\Study\univer\thesis\libraries\datasets\rcv1_test.binary ^
 D:\tmp\results_liblinear.txt
pause