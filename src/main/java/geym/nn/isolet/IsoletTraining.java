package geym.nn.isolet;

import geym.nn.mlperceptron.MlPerceptron;
import geym.nn.util.Utils;
import geym.nn.util.isolet.IsoletReader;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.TransferFunctionType;

import java.io.IOException;
import java.util.Arrays;

public class IsoletTraining implements LearningEventListener {

    public static final String IsoletDir="./src/main/resources/isolet";

    public static void main(String[] args) throws IOException {
        new IsoletTraining().run();
    }

    public void run() throws IOException{
        DataSet trainingSet= IsoletReader.trainingData(IsoletDir+"/isolet1+2+3+4.data");
        int inputCount=trainingSet.getRowAt(0).getInput().length;

        MlPerceptron myMlPerceptron=new MlPerceptron(TransferFunctionType.SIGMOID, inputCount,100,26);
        BackPropagation learningRule=myMlPerceptron.getLearningRule();
        learningRule.setLearningRate(0.05);
        learningRule.setMaxError(0.04d);
        learningRule.setMaxIterations(100);
        learningRule.addListener(this);

        System.out.println("Training neural network...");
        myMlPerceptron.learn(trainingSet);

        // test perceptron
        System.out.println("Testing trained neural network");
        testNeuralNetwork(myMlPerceptron);
    }

    public static void testNeuralNetwork(NeuralNetwork neuralNet) throws IOException{
        DataSet testDataSet=IsoletReader.trainingData(IsoletDir+"/isolet5.data");

        int rightCount=0;
        int i=0;
        for (; i<testDataSet.size();i++){
            neuralNet.setInput(testDataSet.getRowAt(i).getInput());
            neuralNet.calculate();
            double[] networkOutput=neuralNet.getOutput();
            networkOutput= Utils.competiton(networkOutput);

            if (Arrays.equals(networkOutput, testDataSet.getRowAt(i).getDesiredOutput())){
                rightCount++;
            }
            if (1%1000==0){
                System.out.println("正确率： "+rightCount*1.0/(i+1));
            }
        }
        System.out.println("最终正确率："+rightCount*1.0/(i+1));
    }

    public void handleLearningEvent(LearningEvent event) {
        BackPropagation bp=(BackPropagation)event.getSource();
        System.out.println(bp.getCurrentIteration()+". iteration: "+bp.getTotalNetworkError());
    }
}
