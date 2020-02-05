package geym.nn.perceptron;

import geym.nn.perceptron.rule.PerceptronLearningRule;
import org.neuroph.core.Layer;
import org.neuroph.nnet.comp.neuron.InputNeuron;
import org.neuroph.util.*;

public class SimplePerceptron2 extends SimplePerceptron {

    public SimplePerceptron2(int inputNeuronsCount) {
        super(inputNeuronsCount);
    }

    @Override
    protected void createNetWork(int inputNeuronsCount) {
        this.setNetworkType(NeuralNetworkType.PERCEPTRON);

        NeuronProperties inputNeuronProperties=new NeuronProperties();
        inputNeuronProperties.setProperty("neuronType", InputNeuron.class);

        Layer inputLayer= LayerFactory.createLayer(inputNeuronsCount,inputNeuronProperties);
        this.addLayer(inputLayer);

        NeuronProperties outputNeuronProperties=new NeuronProperties();
        outputNeuronProperties.setProperty("transferFunction", TransferFunctionType.STEP);

        Layer outputLayer=LayerFactory.createLayer(2,outputNeuronProperties);
        this.addLayer(outputLayer);

        ConnectionFactory.fullConnect(inputLayer,outputLayer);
        NeuralNetworkFactory.setDefaultIO(this);

        this.setLearningRule(new PerceptronLearningRule());
    }
}
