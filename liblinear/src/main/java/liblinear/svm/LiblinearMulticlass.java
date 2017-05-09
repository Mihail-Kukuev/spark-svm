package liblinear.svm;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import mllib.svm.AppUtils;
import mllib.svm.SVMUtils;
import scala.Tuple2;

import java.io.PrintStream;

public class LiblinearMulticlass extends LiblinearClassifier {

    public LiblinearMulticlass(JavaSparkContext sc) {
        this.sc = sc;
    }

    public static void main(String[] args) {
        AppUtils.checkArguments(args, 2);

        JavaSparkContext sc = AppUtils.initSpark();

        LiblinearMulticlass classifier = new LiblinearMulticlass(sc);
        classifier.setTrainPath(args[0]);
        classifier.setTestPath(args[1]);
        classifier.setEvaluationPath(args[2]);

        classifier.classify();

        sc.stop();
    }

    @Override
    protected void evaluate(JavaRDD<Tuple2<Object, Object>> predictedData, PrintStream ps) {
        SVMUtils.evaluateMulticlass(predictedData, ps);
    }
}
