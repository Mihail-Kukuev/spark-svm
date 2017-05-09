package mllib.svm;

import java.io.PrintStream;

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

    public void printInterval(String message, PrintStream os) {
        t1 = t2;
        startCount();
        System.out.println(message + ": " + (t2 - t1) * 1.0 / 1000);
        os.println(message + ": " + (t2 - t1) * 1.0 / 1000);
    }
}
