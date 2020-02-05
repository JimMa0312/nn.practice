package geym.nn.perceptron;

import geym.nn.perceptron.rule.PerceptronLearningRule;
import org.neuroph.core.Layer;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.nnet.comp.neuron.BiasNeuron;
import org.neuroph.nnet.comp.neuron.InputNeuron;
import org.neuroph.util.*;

public class SimplePerceptron extends NeuralNetwork {
    private static final long serialVersionUID=1L;

    public SimplePerceptron(int inputNeuronsCount){
        this.createNetWork(inputNeuronsCount);
    }

    private void createNetWork(int inputNeuronsCount) {
        this.setNetworkType(NeuralNetworkType.PERCEPTRON);

        NeuronProperties inputNeuronProperties=new NeuronProperties();
        inputNeuronProperties.setProperty("neuronType", InputNeuron.class);

        Layer inputLayer= LayerFactory.createLayer(inputNeuronsCount,inputNeuronProperties);
        this.addLayer(inputLayer);

        inputLayer.addNeuron(new BiasNeuron());

        NeuronProperties outputNeuronProperties=new NeuronProperties();
        outputNeuronProperties.setProperty("transferFunction", TransferFunctionType.STEP);

        Layer outputLayer=LayerFactory.createLayer(1,outputNeuronProperties);
        this.addLayer(outputLayer);

        ConnectionFactory.fullConnect(inputLayer,outputLayer);

        NeuralNetworkFactory.setDefaultIO(this);

        this.setLearningRule(new PerceptronLearningRule());
    }
}
