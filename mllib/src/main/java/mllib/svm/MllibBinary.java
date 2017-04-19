package mllib.svm;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.SVMModel;
import org.apache.spark.mllib.classification.SVMWithSGD;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;

import scala.Tuple2;

public class MllibBinary extends AbstractSVMClassifier {

    private int numIterations;

    public MllibBinary(JavaSparkContext sc) {
        super();
    }

    public static void main(String[] args) {
        AppUtils.checkArguments(args, 3);

        JavaSparkContext sc = AppUtils.initSpark();

        MllibBinary classifier = new MllibBinary(sc);
        classifier.setTrainPath(args[0]);
        classifier.setTestPath(args[1]);
        classifier.setEvaluationPath("D:/tmp/results_mllib.txt");
        classifier.setNumIterations(Integer.valueOf(args[2]));

        classifier.classify();

        sc.stop();
    }

    @Override
    public void classify() {
        timer.startCount();
        JavaRDD<LabeledPoint> loadedTrainData = MLUtils.loadLibSVMFile(sc.sc(), trainPath).toJavaRDD();
        JavaRDD<LabeledPoint> trainingData = SVMUtils.transformBinaryLabels(loadedTrainData);
        trainingData.cache();
        //timer.printInterval(LOADING_TRAINING_TIME);

        SVMModel model = SVMWithSGD.train(trainingData.rdd(), numIterations);
        //model.clearThreshold();
        timer.printInterval(TRAINING_TIME);

        JavaRDD<LabeledPoint> loadedTestData = MLUtils.loadLibSVMFile(sc.sc(), testPath).toJavaRDD();
        JavaRDD<LabeledPoint> transformedTestData = SVMUtils.transformBinaryLabels(loadedTestData);
        //timer.printInterval(LOADING_TEST_TIME);
        JavaRDD<Tuple2<Object, Object>> predictedData = transformedTestData.map(
            p -> new Tuple2<>(model.predict(p.features()), p.label()));
        timer.printInterval(PREDICTION_TIME);

        SVMUtils.evaluateBinary(predictedData, "D:/tmp/results_mllib.txt");
        timer.printInterval(EVALUATION_TIME);
    }

    public int getNumIterations() {
        return numIterations;
    }

    public void setNumIterations(int numIterations) {
        this.numIterations = numIterations;
    }
}