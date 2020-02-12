package geym.nn.mlperceptron.demo;

import com.sun.xml.internal.ws.util.xml.CDATA;
import geym.nn.mlperceptron.MIPerceptronAndOutputNoLearn;
import geym.nn.mlperceptron.MlPerceptron;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.util.TransferFunctionType;

import java.util.Arrays;

public class XorMlPerceptronAndSimple {

    public static void main(String[] args) {
        new XorMlPerceptronAndSimple().run();
    }

    public void run(){
        DataSet trainingSet=new DataSet(2,1);
        trainingSet.add(new DataSetRow(new double[]{0,0}, new double[]{Double.NaN}));
        trainingSet.add(new DataSetRow(new double[]{0,1}, new double[]{Double.NaN}));
        trainingSet.add(new DataSetRow(new double[]{1,0}, new double[]{Double.NaN}));
        trainingSet.add(new DataSetRow(new double[]{1,1}, new double[]{Double.NaN}));

        MlPerceptron myPercetron=new MIPerceptronAndOutputNoLearn(TransferFunctionType.STEP,2,2,1);

        for (DataSetRow testSetRow :
                trainingSet.getRows()) {
            myPercetron.setInput(testSetRow.getInput());
            myPercetron.calculate();
            double[] networkOutput=myPercetron.getOutput();

            System.out.print("Input: "+ Arrays.toString(testSetRow.getInput()));
            System.out.println(" Output: "+Arrays.toString(networkOutput));
        }
    }
}
