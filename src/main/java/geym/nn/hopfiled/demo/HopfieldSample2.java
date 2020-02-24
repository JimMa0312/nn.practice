package geym.nn.hopfiled.demo;

import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.Hopfield;

import java.util.Arrays;

public class HopfieldSample2 {
    public static void main(String[] args) {
        DataSet trainingSet=new DataSet(9);
        trainingSet.add(new DataSetRow(new double[]{1, 0, 1, 1, 1, 1, 1, 0, 1})); // H letter
        trainingSet.add(new DataSetRow(new double[]{1, 1, 1, 0, 1, 0, 0, 1, 0})); // T letter

// create hopfield network
        Hopfield myHopfield = new Hopfield(9);
// learn the training set
        myHopfield.learn(trainingSet);

// test hopfield network
        System.out.println("Testing network");

// add one more 'incomplete' H pattern for testing - it will be recognized as H
        trainingSet.add(new DataSetRow(new double[]{1, 0, 0, 1, 0, 1, 1, 0, 1})); // incomplete H letter

        for(DataSetRow dataRow : trainingSet.getRows()) {

            myHopfield.setInput(dataRow.getInput());
            myHopfield.calculate();
            myHopfield.calculate();
            double[ ] networkOutput = myHopfield.getOutput();

            System.out.print("Input: " + Arrays.toString(dataRow.getInput()) );
            System.out.println(" Output: " + Arrays.toString(networkOutput) );

        }

    }
}
