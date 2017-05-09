package mllib.svm;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.SVMModel;
import org.apache.spark.mllib.classification.SVMWithSGD;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;

import scala.Tuple2;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;

public class MllibBinary extends AbstractSVMClassifier {

    private int numIterations;

	public MllibBinary() {
    }
	
    public MllibBinary(JavaSparkContext sc) {
        this.sc = sc;
    }

    public static void main(String[] args) {
        AppUtils.checkArguments(args, 4);

        JavaSparkContext sc = AppUtils.initSpark();

        MllibBinary classifier = new MllibBinary(sc);
        classifier.setTrainPath(args[0]);
        classifier.setTestPath(args[1]);
        classifier.setEvaluationPath(args[2]);
        classifier.setNumIterations(Integer.valueOf(args[3]));

        classifier.classify();

        sc.stop();
    }

    @Override
    public void classify() {
        PrintStream ps = getPrintStream();
        timer.startCount();

        JavaRDD<LabeledPoint> loadedTrainData = MLUtils.loadLibSVMFile(sc.sc(), trainPath).toJavaRDD();
        JavaRDD<LabeledPoint> trainingData = SVMUtils.transformBinaryLabels(loadedTrainData);
        trainingData.cache();
        //timer.printInterval(LOADING_TRAINING_TIME, ps);

        SVMModel model = SVMWithSGD.train(trainingData.rdd(), numIterations);
        //model.clearThreshold();

        timer.printInterval(TRAINING_TIME, ps);

        JavaRDD<LabeledPoint> loadedTestData = MLUtils.loadLibSVMFile(sc.sc(), testPath).toJavaRDD();
        JavaRDD<LabeledPoint> transformedTestData = SVMUtils.transformBinaryLabels(loadedTestData);
        //timer.printInterval(LOADING_TEST_TIME, ps);
        JavaRDD<Tuple2<Object, Object>> predictedData = transformedTestData.map(
            p -> new Tuple2<>(model.predict(p.features()), p.label()));
        timer.printInterval(PREDICTION_TIME, ps);

        SVMUtils.evaluateBinary(predictedData, ps);
        timer.printInterval(EVALUATION_TIME, ps);
    }

    public int getNumIterations() {
        return numIterations;
    }

    public void setNumIterations(int numIterations) {
        this.numIterations = numIterations;
    }
}