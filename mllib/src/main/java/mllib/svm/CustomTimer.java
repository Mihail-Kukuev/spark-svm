package mllib.svm;

public class CustomTimer {
    private long t1;
    private long t2;


    public CustomTimer() {
        t2 = System.currentTimeMillis();
    }

    public void startCount() {
        t2 = System.currentTimeMillis();
    }

    public void printInterval(String message) {
        t1 = t2;
        startCount();
        System.out.println(message + ": " + (t2 - t1) * 1.0 / 1000);
    }
}
