package geym.nn.perceptron.demo;

import geym.nn.perceptron.SimplePerceptron;
import geym.nn.perceptron.rule.PerceptronLearningRule;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.core.learning.IterativeLearning;

import java.util.Arrays;

public class XorPerceptron implements LearningEventListener {

    public static void main(String[] args) {
        new XorPerceptron().run();
    }

    public void run(){
        DataSet trainingSet=new DataSet(2,1);

        trainingSet.add(new DataSetRow(new double[]{0,0},new double[]{0}));
        trainingSet.add(new DataSetRow(new double[]{0,1},new double[]{1}));
        trainingSet.add(new DataSetRow(new double[]{1,0},new double[]{1}));
        trainingSet.add(new DataSetRow(new double[]{1,1},new double[]{0}));

        SimplePerceptron myPerceptron=new SimplePerceptron(2);

        PerceptronLearningRule learningRule=(PerceptronLearningRule)myPerceptron.getLearningRule();
        learningRule.addListener(this);

        System.out.println("Testing neural network");
        myPerceptron.learn(trainingSet);

        System.out.println("Testing trained neural network");
        testNeuralNetwork(myPerceptron,trainingSet);
    }

    public static void testNeuralNetwork(NeuralNetwork neuralNet, DataSet testSet){
        for (DataSetRow testSetRow:testSet.getRows()){
            neuralNet.setInput(testSetRow.getInput());
            neuralNet.calculate();
            double[] networkOutput=neuralNet.getOutput();

            System.out.print("Input: "+ Arrays.toString(testSetRow.getInput()));
            System.out.println("Output: "+Arrays.toString(networkOutput));
        }
    }

    public void handleLearningEvent(LearningEvent event) {
        IterativeLearning bp=(IterativeLearning)event.getSource();

        System.out.println("iterate:"+bp.getCurrentIteration());
        System.out.println(Arrays.toString(bp.getNeuralNetwork().getWeights()));
    }
}
