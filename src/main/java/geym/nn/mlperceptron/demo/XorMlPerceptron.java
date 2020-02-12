package geym.nn.mlperceptron.demo;

import geym.nn.mlperceptron.MlPerceptron;
import geym.nn.mlperceptron.MlPerceptronBinOutput;
import org.neuroph.core.Connection;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.Neuron;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.core.learning.IterativeLearning;
import org.neuroph.core.learning.LearningRule;
import org.neuroph.util.TransferFunctionType;

import java.util.Arrays;

public class XorMlPerceptron implements LearningEventListener {

    public static void main(String[] args) {
        new XorMlPerceptron().run();
    }

    public void run(){
        DataSet trainingSet=new DataSet(2,1);
        trainingSet.add(new DataSetRow(new double[]{0,0}, new double[]{0}));
        trainingSet.add(new DataSetRow(new double[]{0,1}, new double[]{1}));
        trainingSet.add(new DataSetRow(new double[]{1,0}, new double[]{1}));
        trainingSet.add(new DataSetRow(new double[]{1,1}, new double[]{0}));

        MlPerceptron myPerceptron=new MlPerceptronBinOutput(TransferFunctionType.SIGMOID,2,4,1);
        LearningRule learningRule=myPerceptron.getLearningRule();
        learningRule.addListener(this);

        System.out.println("Traning neural network...");
        myPerceptron.learn(trainingSet);
        System.out.println("Testing trained neural netwoek...");
        testNeuralNetwork(myPerceptron,trainingSet);
    }

    public static void testNeuralNetwork(NeuralNetwork neuralNetwork, DataSet testSet){
        for (DataSetRow testSetRow:testSet.getRows()){
            neuralNetwork.setInput(testSetRow.getInput());
            neuralNetwork.calculate();
            double[] networkOutput=neuralNetwork.getOutput();

            System.out.print("Input: "+ Arrays.toString(testSetRow.getInput()));
            System.out.println(" Output: "+Arrays.toString(networkOutput));
        }
    }

    public void handleLearningEvent(LearningEvent event) {
        System.out.println("=================================");
        System.out.println(event.getClass().toString());
        IterativeLearning bp=(IterativeLearning)event.getSource();
        System.out.println("iterate:"+bp.getCurrentIteration());
        Neuron neuron=(Neuron)bp.getNeuralNetwork().getOutputNeurons().get(0);
        for (Connection conn :
                neuron.getInputConnections()) {
            System.out.println(conn.getWeight().value);
        }
    }
}
