package geym.nn.adaline;

import org.neuroph.core.Layer;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.nnet.comp.neuron.BiasNeuron;
import org.neuroph.nnet.comp.neuron.InputNeuron;
import org.neuroph.nnet.learning.LMS;
import org.neuroph.util.*;

public class Adaline extends NeuralNetwork<LMS> {
    private static final long serialVersionUID=1L;

    public Adaline(int inputNeuronsCount, int outputNeuronsCount) {
        this.crateNetwork(inputNeuronsCount,outputNeuronsCount);
    }

    private void crateNetwork(int inputNeuronsCount, int outputNeuronCount){
        this.setNetworkType(NeuralNetworkType.ADALINE);

        NeuronProperties inNeuronProperties=new NeuronProperties();
        inNeuronProperties.setProperty("neuronType", InputNeuron.class);
        /**
         * 源书代码在这里使用的是未设定输入函数，直接函数后跟随Line输出函数，因此在2.96运行时架构没有将Input值合理注入，会导致学习异常
         * 因为如果使用普通的Neuron类，程序在执行cacule函数时，会根据与该神经元输入线路进行权重累加和，第一层（即输入层）之前是没有结点输入的，所以输入线路的个数为0，因此Cacule计算时，就会不进行计算，返回0
         * InputNeuron类将cacule函数进行了重写，不进行计算，直接将DataSet的input值复制到totalinput中，
         */
//        inNeuronProperties.setProperty("transferFunction", TransferFunctionType.LINEAR);
        Layer inputLayer= LayerFactory.createLayer(inputNeuronsCount,inNeuronProperties);
        inputLayer.addNeuron(new BiasNeuron());

        this.addLayer(inputLayer);

        NeuronProperties outNeuronProperties=new NeuronProperties();
        outNeuronProperties.setProperty("transferFunction",TransferFunctionType.LINEAR);
        Layer outputLayer=LayerFactory.createLayer(outputNeuronCount, outNeuronProperties);
        this.addLayer(outputLayer);

        ConnectionFactory.fullConnect(inputLayer,outputLayer);

        NeuralNetworkFactory.setDefaultIO(this);

        LMS lms=new LMS();

        lms.setLearningRate(0.05);
        lms.setMaxError(0.5);
        lms.setBatchMode(true);
        this.setLearningRule(lms);
    }
}
