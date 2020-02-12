package geym.nn.mlperceptron.demo;

import geym.nn.mlperceptron.MlPerceptron;
import geym.nn.util.Utils;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.core.learning.LearningRule;
import org.neuroph.core.learning.SupervisedLearning;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.TransferFunctionType;

import java.util.Random;

public class ParityCheck implements LearningEventListener {
    public static void main(String[] args) {
        new ParityCheck().run();
    }

    public void run(){
        DataSet traingSet=new DataSet(32,4);
        for (int i = 0; i < 2000; i++) {
            int in=new Random().nextInt();
            traingSet.add(new DataSetRow(int2double(in),int2prop(in)));
        }

        MlPerceptron myMlPerceptron=new MlPerceptron(TransferFunctionType.SIGMOID, 32,10,4);

        LearningRule learningRule=myMlPerceptron.getLearningRule();
        learningRule.addListener(this);

        ((SupervisedLearning)learningRule).setMaxError(0.0001d);

        System.out.println("Training neural network...");
        myMlPerceptron.learn(traingSet);

        System.out.println("Testing trained neural network...");
        testNeuralNetwork(myMlPerceptron);
    }

    public static void testNeuralNetwork(NeuralNetwork neuralNetwork){
        int badcount=0;
        int COUNT=50000;

        for (int i = 0; i < COUNT; i++) {
            int in=new Random().nextInt();
            double[] inputnumber=int2double(in);
            neuralNetwork.setInput(inputnumber);
            neuralNetwork.calculate();
            double[] networkOutput=neuralNetwork.getOutput();
            networkOutput= Utils.competiton(networkOutput);
            String networkOutputDisplay=networkOutputDisplay(networkOutput);
            String cc=correctClassify(in);
            System.out.print(in+" "+networkOutputDisplay+" ");
            if (i%50==0){
                System.out.println();
            }

            if (!cc.equals(networkOutputDisplay)){
                badcount++;
                System.out.print("判别错误: "+in);
                System.out.print(" correctClassify="+cc);
                System.out.println(" networkOutputDisplay="+networkOutputDisplay);
            }
        }

        System.out.println();
        System.out.println("正确率： " + ((COUNT-badcount*1.0)/COUNT*100.0) +"%");
    }

    public static double[] int2double(int i){
        double[] re=new double[32];

        for (int j=0;j<32;j++){
            re[j]=(double)((i>>j)&0x01);
        }

        return re;
    }

    public static String networkOutputDisplay(double[] networkOutput){
        if ((int)networkOutput[3]==1) return "正偶数";
        if ((int)networkOutput[2]==1) return "负偶数";
        if ((int)networkOutput[1]==1) return "正奇数";
        if ((int)networkOutput[0]==1) return "负奇数";

        return "未知";
    }

    public static String correctClassify(int i){
        if (i > 0 && i%2==0){
            return "正偶数";
        }
        else if(i <0 && i%2==0){
            return "负偶数";
        }else if(i>0 && i%2!=0){
            return "正奇数";
        }else if(i<0 && i%2!=0){
            return "负奇数";
        }

        return "0";
    }

    public static double[] int2prop(int i){
        //正偶数
        double[] pe={0,0,0,1};
        //负偶数
        double[] ne={0,0,1,0};
        //正奇数
        double[] po={0,1,0,0};
        //负奇数
        double[] no={1,0,0,0};

        if (i > 0 && i%2==0){
            return pe;
        }
        else if(i <0 && i%2==0){
            return ne;
        }else if(i>0 && i%2!=0){
            return po;
        }else if(i<0 && i%2!=0){
            return no;
        }

        return pe;
    }

    public void handleLearningEvent(LearningEvent event) {
        BackPropagation bp=(BackPropagation)event.getSource();
        System.out.println(bp.getCurrentIteration() +". iteration: "+bp.getTotalNetworkError());
    }
}
