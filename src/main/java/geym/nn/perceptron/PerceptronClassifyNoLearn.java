package geym.nn.perceptron;

import org.neuroph.core.Layer;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.Neuron;
import org.neuroph.nnet.comp.neuron.InputNeuron;
import org.neuroph.util.*;

import java.util.Arrays;
import java.util.Scanner;

/**
 * 识别四个象限，因为判断边界互相垂直，因此 向量使用为[1,0],[0,1]
 *
 */
public class PerceptronClassifyNoLearn extends NeuralNetwork {
    /**
     * 本类迭代版本
     */
    private static final long serialVersionUID=1L;

    /**
     * 创建一个新的感知机神经网络，输入点的数量
     * @param inputNeuronsCount 输入点数量
     */
    public PerceptronClassifyNoLearn(int inputNeuronsCount){
        this.createNetWork(inputNeuronsCount);
    }

    private void createNetWork(int inputNeuronsCount) {
        this.setNetworkType(NeuralNetworkType.PERCEPTRON);

        //设置输入神经元的配置
        NeuronProperties inputNeuronProperties=new NeuronProperties();
        inputNeuronProperties.setProperty("neuronType", InputNeuron.class);

        //有输入神经元构成输入层
        Layer inputLayer= LayerFactory.createLayer(inputNeuronsCount,inputNeuronProperties);
        this.addLayer(inputLayer);

        NeuronProperties outputNeuronProperties=new NeuronProperties();
        outputNeuronProperties.setProperty("transferFunction", TransferFunctionType.STEP);
        Layer outputLayer=LayerFactory.createLayer(2,outputNeuronProperties);
        this.addLayer(outputLayer);

        ConnectionFactory.fullConnect(inputLayer,outputLayer);
        NeuralNetworkFactory.setDefaultIO(this);

        Neuron n=outputLayer.getNeuronAt(0);

        n.getInputConnections().get(0).getWeight().setValue(1);
        n.getInputConnections().get(1).getWeight().setValue(0);
        n=outputLayer.getNeuronAt(1);
        n.getInputConnections().get(0).getWeight().setValue(0);
        n.getInputConnections().get(1).getWeight().setValue(1);
    }

    public static String posToString(double[] networkOutput){
        double add=networkOutput[0]+networkOutput[1];
        double less=networkOutput[0]-networkOutput[1];

        if (add==2){
            return "第一象限";
        }else if (add==0){
            return "第三象限";
        }else if (less==1){
            return "第四象限";
        }else{
            return "第二象限";
        }
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        String line=null;
        double[] input=new double[2];

        PerceptronClassifyNoLearn perceptron=new PerceptronClassifyNoLearn(2);
        try {
            while((line=scanner.nextLine()) != null){
                String[] numbers=line.split("[\\s|,|;]");
                input[0]=Double.parseDouble(numbers[0]);
                input[1]=Double.parseDouble(numbers[1]);

                perceptron.setInput(input);
                perceptron.calculate();
                double[] networkOutput=perceptron.getOutput();
                System.out.println(Arrays.toString(input) +"="+posToString(networkOutput)+Arrays.toString(networkOutput));
            }
        }finally {
            scanner.close();
        }
    }
}
