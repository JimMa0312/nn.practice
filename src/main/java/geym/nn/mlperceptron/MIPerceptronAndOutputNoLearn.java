package geym.nn.mlperceptron;

import org.neuroph.core.Connection;
import org.neuroph.core.Layer;
import org.neuroph.core.Neuron;
import org.neuroph.core.Weight;
import org.neuroph.core.input.And;
import org.neuroph.core.transfer.Linear;
import org.neuroph.nnet.comp.neuron.BiasNeuron;
import org.neuroph.nnet.comp.neuron.InputNeuron;
import org.neuroph.util.*;

import java.util.List;

public class MIPerceptronAndOutputNoLearn extends MlPerceptron {
    public MIPerceptronAndOutputNoLearn(TransferFunctionType transferFunctionType, int... neuronsInLayers) {
        super(transferFunctionType, neuronsInLayers);
    }

    @Override
    protected void createNetWork(List<Integer> neuronsInLayers, NeuronProperties neuronProperties) {
        this.setNetworkType(NeuralNetworkType.MULTI_LAYER_PERCEPTRON);

        NeuronProperties inputNeuronProperties = new NeuronProperties(InputNeuron.class, Linear.class);
        Layer layer = LayerFactory.createLayer(neuronsInLayers.get(0), inputNeuronProperties);
        layer.addNeuron(new BiasNeuron());

        this.addLayer(layer);

        Layer prevLayer = layer;
        int layerIdx;
        for (layerIdx = 1; layerIdx < neuronsInLayers.size() - 1; layerIdx++) {
            Integer neuronsNum = neuronsInLayers.get(layerIdx);
            layer = LayerFactory.createLayer(neuronsNum, neuronProperties);
            layer.addNeuron(new BiasNeuron());
            this.addLayer(layer);
            ConnectionFactory.fullConnect(prevLayer, layer);

            prevLayer = layer;
        }


        Neuron n1=layer.getNeuronAt(0);
        List<Connection> c1=n1.getInputConnections();
        c1.get(0).setWeight(new Weight(2));
        c1.get(1).setWeight(new Weight(2));
        c1.get(2).setWeight(new Weight(-1));

        Neuron n2=layer.getNeuronAt(1);
        List<Connection> c2=n2.getInputConnections();
        c2.get(0).setWeight(new Weight(-2));
        c2.get(1).setWeight(new Weight(-2));
        c2.get(2).setWeight(new Weight(3));

        //建立输出层
        Integer neuronsNum=neuronsInLayers.get(layerIdx);
        NeuronProperties outProperties=new NeuronProperties();
        outProperties.put("inputFunction", And.class);
        layer=LayerFactory.createLayer(neuronsNum,outProperties);
        this.addLayer(layer);
        ConnectionFactory.fullConnect(prevLayer,layer);
        prevLayer=layer;

        NeuralNetworkFactory.setDefaultIO(this);
    }
}
