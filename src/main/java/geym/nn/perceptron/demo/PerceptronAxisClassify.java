package geym.nn.perceptron.demo;

import geym.nn.perceptron.SimplePerceptron2;
import geym.nn.perceptron.rule.PerceptronLearningRule;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.core.learning.IterativeLearning;

import java.util.Arrays;
import java.util.Random;

public class PerceptronAxisClassify extends NeuralNetwork implements LearningEventListener {

    private static final long serialVersionUID=7685380676792200464L;

    static Random r=new Random();

    public static void main(String[] args) {
        new PerceptronAxisClassify().run();
    }

    public static double nextDouble(){
        double re=0;
        while ((re=r.nextDouble())!=0){
            return re;
        }

        return r.nextDouble();
    }

    public void run(){
        DataSet td=new DataSet(2,2);
        for (int i=0;i<10000;i++){
            td.add(new DataSetRow(new double[]{1*nextDouble(),1*nextDouble()},new double[]{1,1}));
            td.add(new DataSetRow(new double[]{-1*nextDouble(),1*nextDouble()},new double[]{0,1}));
            td.add(new DataSetRow(new double[]{-1*nextDouble(),-1*nextDouble()},new double[]{0,0}));
            td.add(new DataSetRow(new double[]{1*nextDouble(),-1*nextDouble()},new double[]{1,0}));
        }

        SimplePerceptron2 myPerceptron=new SimplePerceptron2(2);

        PerceptronLearningRule learningRule = (PerceptronLearningRule)myPerceptron.getLearningRule();
        learningRule.setMaxError(0.001);
        learningRule.addListener(this);

        System.out.println("Training neural network...");
        myPerceptron.learn(td);

        System.out.println("Testing trained neural network...");
        testNeuralNetwork(myPerceptron);
    }

    public static void testNeuralNetwork(NeuralNetwork neuralNet){
        DataSet td=new DataSet(2,2);

        for (int i=0; i<100; i++){
            td.add(new DataSetRow(new double[]{1*nextDouble(),1*nextDouble()},new double[]{1,1}));
            td.add(new DataSetRow(new double[]{-1*nextDouble(),1*nextDouble()},new double[]{0,1}));
            td.add(new DataSetRow(new double[]{-1*nextDouble(),-1*nextDouble()},new double[]{0,0}));
            td.add(new DataSetRow(new double[]{1*nextDouble(),-1*nextDouble()},new double[]{1,0}));
        }

        int correctCount=0;
        int incorrectCount=0;
        for (DataSetRow testSetRow: td.getRows()){
            neuralNet.setInput((testSetRow.getInput()));
            neuralNet.calculate();
            double[] networkOutput=neuralNet.getOutput();
            if (Arrays.equals(networkOutput, testSetRow.getDesiredOutput())){
                correctCount++;
            }else{
                incorrectCount++;
            }
        }
        System.out.println("正确率："+(correctCount*1.0 /(correctCount+incorrectCount)));
    }

    public void handleLearningEvent(LearningEvent event) {
        IterativeLearning bp=(IterativeLearning)event.getSource();
        System.out.println("iterate: "+bp.getCurrentIteration());
        System.out.print("TotalNetworkError: ");
        System.out.println(((PerceptronLearningRule)bp.getNeuralNetwork().getLearningRule()).getTotalNetworkError());
    }
}
