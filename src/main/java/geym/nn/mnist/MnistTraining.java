package geym.nn.mnist;

import geym.nn.mlperceptron.MlPerceptron;
import geym.nn.util.Utils;
import geym.nn.util.mnist.MnistReader;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.TransferFunctionType;

import java.io.IOException;
import java.util.Arrays;

public class MnistTraining implements LearningEventListener {
    public static final String MnistDir="src/main/resources/handswriter/mnist";

    public static void main(String[] args) throws IOException {
        new MnistTraining().run();
    }

    public void run() throws IOException{
        DataSet trainingDataSet= MnistReader.trainingData(MnistDir+"/train-images-idx3-ubyte",
                MnistDir+"/train-labels-idx1-ubyte",10000);

        MlPerceptron myMlperceptron=new MlPerceptron(TransferFunctionType.SIGMOID, 784,100,10);

        BackPropagation learningRule=myMlperceptron.getLearningRule();
        learningRule.setLearningRate(0.05);
        learningRule.setMaxError(0.001d);
        learningRule.setMaxIterations(10);
        learningRule.addListener(this);

        System.out.println("Training nerual network...");
        myMlperceptron.learn(trainingDataSet);
        System.out.println("Testing trained neural network...");
        testNerualNetwork(myMlperceptron);
    }

    public static void testNerualNetwork(NeuralNetwork neuralNet)throws IOException{
        DataSet testDataSet= MnistReader.trainingData(MnistDir+"/t10k-images-idx3-ubyte",
                MnistDir+"/t10k-labels-idx1-ubyte",10000);

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
            if(i%100==0){
                System.out.println("正确率："+rightCount*1.0/(i+1)) ;
            }
        }

        System.out.println("正确率： "+rightCount*1.0/(i+1));
    }

    public void handleLearningEvent(LearningEvent event) {
        BackPropagation bp=(BackPropagation)event.getSource();
        System.out.println(bp.getCurrentIteration()+". iteration: "+bp.getTotalNetworkError());
    }
}
