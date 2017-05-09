package mllib.svm;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintStream;
import java.io.PrintWriter;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.regression.LabeledPoint;

import scala.Tuple2;

public class SVMUtils {

    public static void evaluateBinary(JavaRDD<Tuple2<Object, Object>> predictedData, PrintStream ps) {
        predictedData.cache();

        double truePredictions = predictedData.filter(tuple -> tuple._1$mcD$sp() == tuple._2$mcD$sp()).count();
        double accuracy = truePredictions / predictedData.count();

        BinaryClassificationMetrics metrics = new BinaryClassificationMetrics(predictedData.rdd());

        String output = "Accuracy: " + accuracy
            + "\nPrecision: " + metrics.precisionByThreshold().first()
            + "\nRecall: " + metrics.recallByThreshold().first()
            + "\nF-measure: " + metrics.fMeasureByThreshold().first();

        System.out.println(output);
        ps.println(output);
    }

    public static void evaluateMulticlass(JavaRDD<Tuple2<Object, Object>> predictedData, PrintStream ps) {
        predictedData.cache();
        MulticlassMetrics metrics = new MulticlassMetrics(predictedData.rdd());

        String output = "Precision: " + metrics.weightedPrecision()
            + "\nRecall: " + metrics.weightedRecall()
            + "\nF-measure: " + metrics.weightedFMeasure()
            + "\nTrue Positive Rate: " + metrics.weightedFalsePositiveRate();

        System.out.println(output);
        ps.println(output);
    }

    private static void printToFile(String s, String filename) {
        try (PrintWriter pw = new PrintWriter(new File(filename))) {
            pw.println(s);
        }
        catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    static JavaRDD<LabeledPoint> transformBinaryLabels(JavaRDD<LabeledPoint> data) {
        return data.map(labeledPoint ->
            labeledPoint.label() == -1 ? new LabeledPoint(0, labeledPoint.features()) : labeledPoint);
    }

    static JavaRDD<LabeledPoint> transformMulticlassLabels(JavaRDD<LabeledPoint> data) {
        return data.map(labeledPoint -> new LabeledPoint(labeledPoint.label() - 1, labeledPoint.features()));
    }
}
