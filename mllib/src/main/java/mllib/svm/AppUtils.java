package mllib.svm;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;

public class AppUtils {

    public static JavaSparkContext initSpark() {
        SparkConf conf = new SparkConf().setAppName("SVMApp");
        return new JavaSparkContext(conf);
    }

    public static void checkArguments(String[] args, int count) {
        if (args.length < count) {
            System.out.println("Parameters for train and test files were not found. Try again, please.");
            System.exit(1);
        }
    }
}
