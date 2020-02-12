package geym.nn.mlperceptron;

import org.neuroph.core.Layer;
import org.neuroph.core.transfer.Linear;
import org.neuroph.nnet.comp.neuron.BiasNeuron;
import org.neuroph.nnet.comp.neuron.InputNeuron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.*;
import org.neuroph.util.random.NguyenWidrowRandomizer;

import java.util.List;

/**
 * 具有2值输出的多层感知机
 */
public class MlPerceptronBinOutput extends MlPerceptron {
    public MlPerceptronBinOutput(TransferFunctionType transferFunctionType, int... neuronsInLayers) {
        super(transferFunctionType, neuronsInLayers);
    }

    @Override
    protected void createNetWork(List<Integer> neuronsInLayers, NeuronProperties neuronProperties) {
        this.setNetworkType(NeuralNetworkType.MULTI_LAYER_PERCEPTRON);

        NeuronProperties inputNeuronProperties=new NeuronProperties(InputNeuron.class, Linear.class);
        Layer layer = LayerFactory.createLayer(neuronsInLayers.get(0), inputNeuronProperties);
        layer.addNeuron(new BiasNeuron());
        this.addLayer(layer);

        Layer prevLayer=layer;

        int layerIdx;
        for (layerIdx=1; layerIdx<neuronsInLayers.size()-1;layerIdx++){
            Integer neuronsNum=neuronsInLayers.get(layerIdx);

            layer=LayerFactory.createLayer(neuronsNum,neuronProperties);
            layer.addNeuron(new BiasNeuron());
            this.addLayer(layer);

            ConnectionFactory.fullConnect(prevLayer, layer);

            prevLayer=layer;
        }

        Integer neuronsNum=neuronsInLayers.get(layerIdx);
        NeuronProperties outputNeuronProperties=new NeuronProperties();
        outputNeuronProperties.setProperty("transferFunction", TransferFunctionType.STEP);
        layer=LayerFactory.createLayer(neuronsNum,outputNeuronProperties);
        this.addLayer(layer);
        ConnectionFactory.fullConnect(prevLayer,layer);

        NeuralNetworkFactory.setDefaultIO(this);

        this.setLearningRule(new BackPropagation());
        this.randomizeWeights(new NguyenWidrowRandomizer(-0.7,0.7));
    }
}
