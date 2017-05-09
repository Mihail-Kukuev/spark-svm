package liblinear.svm;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import mllib.svm.AppUtils;
import mllib.svm.SVMUtils;
import scala.Tuple2;

import java.io.PrintStream;

public class LiblinearBinary extends LiblinearClassifier {

    public LiblinearBinary(JavaSparkContext sc) {
        this.sc = sc;
    }

    public static void main(String[] args) {
        AppUtils.checkArguments(args, 3);

        JavaSparkContext sc = AppUtils.initSpark();

        LiblinearBinary classifier = new LiblinearBinary(sc);
        classifier.setTrainPath(args[0]);
        classifier.setTestPath(args[1]);
        classifier.setEvaluationPath(args[2]);

        classifier.classify();

        sc.stop();
    }

    @Override
    protected void evaluate(JavaRDD<Tuple2<Object, Object>> predictedData, PrintStream ps) {
        SVMUtils.evaluateBinary(predictedData, ps);
    }
}