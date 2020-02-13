package geym.nn.mlperceptron.demo;

import geym.nn.mlperceptron.MlPerceptron;
import geym.nn.util.Utils;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.TransferFunctionType;

import java.io.*;
import java.util.Arrays;
import java.util.List;
import java.util.Vector;

public class AnimalClassify implements LearningEventListener {

    public static void main(String[] args) throws IOException {
        new AnimalClassify().run();
    }

    public static List<DataSetRow> getTrainData(String filepath)throws IOException{
        FileInputStream in=new FileInputStream(filepath);
        BufferedReader br=new BufferedReader(new InputStreamReader(new DataInputStream(in)));

        List<DataSetRow> re=new Vector<DataSetRow>();
        String line;

        int Attribute_Len=16;
        while ((line = br.readLine())!=null){
            String[] item=line.split("\t");
            double[] inputs=new double[Attribute_Len];
            int i;
            for (i=0;i<Attribute_Len;i++){
                inputs[i]=Double.parseDouble(item[i]);
            }

            double[] outputs=new double[7];
            for (; i<Attribute_Len+7;i++){
                outputs[i-Attribute_Len]=Double.parseDouble(item[i]);
            }

            re.add(new DataSetRow(inputs,outputs));
        }

        return re;
    }

    public void run() throws IOException{
        DataSet trainingSet=new DataSet(16,7);
        List<DataSetRow> rows=getTrainData("./src/main/resources/Zoo/zoo.90.percent.txt");

        for (DataSetRow setRow :
                rows) {
            trainingSet.add(setRow);
        }

        MlPerceptron myMlPerceptron=new MlPerceptron(TransferFunctionType.SIGMOID, 16,6,7);
        myMlPerceptron.getLearningRule().setMaxError(0.01d);
        myMlPerceptron.getLearningRule().addListener(this);

        System.out.println("Training neural network...");
        myMlPerceptron.learn(trainingSet);
        System.out.println("Testing trained nrual network...");
        testNeuralNetWork(myMlPerceptron);
    }

    public static void testNeuralNetWork(NeuralNetwork neuralNet) throws IOException{
        List<DataSetRow> rows=getTrainData("./src/main/resources/Zoo/zoo.10.percent.txt");
        int count=0, correctNumber=0;

        for (DataSetRow row :
                rows) {
            neuralNet.setInput(row.getInput());
            neuralNet.calculate();

            double[] netWorkOutput=neuralNet.getOutput();
            netWorkOutput= Utils.competiton(netWorkOutput);
            count++;
            if (Arrays.equals(netWorkOutput,row.getDesiredOutput())){
                correctNumber++;
            }
        }

        System.out.println("判断正确率："+(correctNumber*1.0/count)*100+"%");
    }

    public void handleLearningEvent(LearningEvent event) {
        BackPropagation bp=(BackPropagation)event.getSource();
        System.out.println(bp.getCurrentIteration()+". iteration: "+bp.getTotalNetworkError());
    }
}
