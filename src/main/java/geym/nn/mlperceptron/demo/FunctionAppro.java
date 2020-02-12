package geym.nn.mlperceptron.demo;

import ch.qos.logback.core.util.FileUtil;
import geym.nn.mlperceptron.MlPerceptron;
import geym.nn.mlperceptron.MlPerceptronLineOutput;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.core.learning.LearningRule;
import org.neuroph.core.learning.SupervisedLearning;
import org.neuroph.nnet.learning.BackPropagation;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

public class FunctionAppro implements LearningEventListener {

    public static int i = 8;

    public static void main(String[] args) throws IOException {
        new FunctionAppro().run(16,0.0001d);
    }

    public double func(double x) {
        return 1 + Math.sin(Math.PI * i / 4 * x);
    }

    public void run(int hiddenLayerNums, double maxError) throws IOException {
        DataSet trainingSet=new DataSet(1, 1);
        for (int i=0; i<2000;i++){
            double in=new Random().nextDouble()*4-2;//[-2,2]
            double out=func(in);

            trainingSet.add(new DataSetRow(new double[]{in}, new double[]{out}));
        }

        MlPerceptron myMlPerceptron=new MlPerceptronLineOutput(1,hiddenLayerNums,1);
        myMlPerceptron.setLearningRule(new BackPropagation());
        myMlPerceptron.getLearningRule().setMaxError(maxError);

        LearningRule learningRule=myMlPerceptron.getLearningRule();
        learningRule.addListener(this);
        System.out.println("Training neural network...");
        myMlPerceptron.learn(trainingSet);

        System.out.println("Testing trained neural network...");
        testNerualNetwork(myMlPerceptron,hiddenLayerNums,maxError);
    }

    public static void testNerualNetwork(NeuralNetwork neuralNet, int hiddenLayerNums, double maxError)throws IOException{
        StringBuffer x=new StringBuffer();
        StringBuffer y=new StringBuffer();

        for (int j = 0; j < 100; j++) {
            double in=new Random().nextDouble()*4-2;
            neuralNet.setInput(in);
            neuralNet.calculate();
            double[] networkOutput=neuralNet.getOutput();

            x.append(in+"\t");
            y.append(networkOutput[0]+"\t");
        }

        FileWriter xFile=new FileWriter("src/main/resources/FunctionCorrdinate/x_"+hiddenLayerNums+"_"+maxError+".txt");
        FileWriter yFile=new FileWriter("src/main/resources/FunctionCorrdinate/y_"+hiddenLayerNums+"_"+maxError+".txt");
        xFile.write(x.toString());
        yFile.write(y.toString());

        xFile.close();
        yFile.close();
    }

    public void handleLearningEvent(LearningEvent event) {
        SupervisedLearning bp = (SupervisedLearning) event.getSource();
        System.out.println(bp.getCurrentIteration() + ". iteration : " + bp.getTotalNetworkError());
    }
}
