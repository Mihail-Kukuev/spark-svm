package mllib.svm;

import org.apache.spark.api.java.JavaSparkContext;


public abstract class AbstractSVMClassifier {

    protected static final String LOADING_TRAINING_TIME = "Loading training file time";
    protected static final String LOADING_TEST_TIME = "Loading training file time";
    protected static final String TRAINING_TIME = "Training time";
    protected static final String PREDICTION_TIME = "Prediction time";
    protected static final String EVALUATION_TIME = "Evaluation time";

    protected JavaSparkContext sc;
    protected String trainPath;
    protected String testPath;
    protected String evaluationPath;
    protected CustomTimer timer = new CustomTimer();

    public AbstractSVMClassifier() {
    }

    public AbstractSVMClassifier(JavaSparkContext sc) {
        this.sc = sc;
    }


    public abstract void classify();

    public String getTrainPath() {
        return trainPath;
    }

    public void setTrainPath(String trainPath) {
        this.trainPath = trainPath;
    }

    public String getTestPath() {
        return testPath;
    }

    public void setTestPath(String testPath) {
        this.testPath = testPath;
    }

    public String getEvaluationPath() {
        return evaluationPath;
    }

    public void setEvaluationPath(String evaluationPath) {
        this.evaluationPath = evaluationPath;
    }
}
