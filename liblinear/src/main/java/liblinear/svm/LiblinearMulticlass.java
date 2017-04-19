package liblinear.svm;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import mllib.svm.AppUtils;
import mllib.svm.SVMUtils;
import scala.Tuple2;

public class LiblinearMulticlass extends LiblinearClassifier {

    public LiblinearMulticlass(JavaSparkContext sc) {
        super();
    }

    public static void main(String[] args) {
        AppUtils.checkArguments(args, 2);

        JavaSparkContext sc = AppUtils.initSpark();

        LiblinearMulticlass classifier = new LiblinearMulticlass(sc);
        classifier.setTrainPath(args[0]);
        classifier.setTestPath(args[1]);
        classifier.setEvaluationPath("D:/tmp/results-liblinear-multiclass.txt");

        classifier.classify();

        sc.stop();
    }

    @Override
    protected void evaluate(JavaRDD<Tuple2<Object, Object>> predictedData) {
        SVMUtils.evaluateMulticlass(predictedData, evaluationPath);
    }
}
