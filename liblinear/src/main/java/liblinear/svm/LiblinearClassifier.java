package liblinear.svm;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.rdd.RDD;

import mllib.svm.AbstractSVMClassifier;
import scala.Tuple2;
import tw.edu.ntu.csie.liblinear.DataPoint;
import tw.edu.ntu.csie.liblinear.LiblinearModel;
import tw.edu.ntu.csie.liblinear.SparkLiblinear;
import tw.edu.ntu.csie.liblinear.Utils;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;

public class LiblinearClassifier extends AbstractSVMClassifier {
	
	public LiblinearClassifier() {
    }
	
	public LiblinearClassifier(JavaSparkContext sc) {
        this.sc = sc;
    }

    @Override
    public void classify() {
	    PrintStream ps = getPrintStream();
        timer.startCount();
        RDD<DataPoint> trainingData = Utils.loadLibSVMData(sc.sc(), trainPath);
        trainingData.cache();
        //timer.printInterval(LOADING_TRAIN_TIME, ps);

        LiblinearModel model = SparkLiblinear.train(trainingData);
        //model.clearThreshold();
        timer.printInterval(TRAINING_TIME, ps);

        JavaRDD<DataPoint> testData = Utils.loadLibSVMData(sc.sc(), testPath).toJavaRDD();
        //timer.printInterval(LOADING_TEST_TIME);
        JavaRDD<Tuple2<Object, Object>> predictedData = testData.map(point -> new Tuple2<>(model.predict(point), point.y()));
        timer.printInterval(PREDICTION_TIME, ps);

        evaluate(predictedData, ps);
        timer.printInterval(EVALUATION_TIME, ps);
    }

    protected void evaluate(JavaRDD<Tuple2<Object, Object>> predictedData, PrintStream ps) {}

}
