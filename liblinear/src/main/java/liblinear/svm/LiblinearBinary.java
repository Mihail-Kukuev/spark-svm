package liblinear.svm;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import mllib.svm.AppUtils;
import mllib.svm.SVMUtils;
import scala.Tuple2;

public class LiblinearBinary extends LiblinearClassifier {

    public LiblinearBinary(JavaSparkContext sc) {
        super();
    }

    public static void main(String[] args) {
        AppUtils.checkArguments(args, 2);

        JavaSparkContext sc = AppUtils.initSpark();

        LiblinearBinary classifier = new LiblinearBinary(sc);
        classifier.setTrainPath(args[0]);
        classifier.setTestPath(args[1]);
        classifier.setEvaluationPath("D:/tmp/results_liblinear.txt");

        classifier.classify();

        sc.stop();
    }

    @Override
    protected void evaluate(JavaRDD<Tuple2<Object, Object>> predictedData) {
        SVMUtils.evaluateBinary(predictedData, evaluationPath);
    }
}