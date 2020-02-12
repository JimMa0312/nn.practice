package geym.nn.mlperceptron;

import org.neuroph.core.Layer;
import org.neuroph.core.transfer.Linear;
import org.neuroph.nnet.comp.neuron.BiasNeuron;
import org.neuroph.nnet.comp.neuron.InputNeuron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.*;
import org.neuroph.util.random.NguyenWidrowRandomizer;

import java.util.List;

public class MlPerceptronLineOutput extends MlPerceptron {
    public MlPerceptronLineOutput(int... neuronsInLayers) {
        super(neuronsInLayers);
    }

    public MlPerceptronLineOutput(TransferFunctionType transferFunctionType, int... neuronsInLayers) {
        super(transferFunctionType, neuronsInLayers);
    }

    @Override
    protected void createNetWork(List<Integer> neuronsInLayers, NeuronProperties neuronProperties) {
        this.setNetworkType(NeuralNetworkType.MULTI_LAYER_PERCEPTRON);

        NeuronProperties inputNeuronProperties=new NeuronProperties(InputNeuron.class, Linear.class);
        Layer layer= LayerFactory.createLayer(neuronsInLayers.get(0), inputNeuronProperties);
        layer.addNeuron(new BiasNeuron());
        this.addLayer(layer);

        Layer prevLayer=layer;

        int layerIdx;

        for (layerIdx=1; layerIdx<neuronsInLayers.size()-1; layerIdx++){
            Integer neuronsNum=neuronsInLayers.get(layerIdx);

            layer=LayerFactory.createLayer(neuronsNum,neuronProperties);
            layer.addNeuron(new BiasNeuron());

            this.addLayer(layer);
            ConnectionFactory.fullConnect(prevLayer,layer);

            prevLayer=layer;
        }

        //get outputLayer
        Integer neuronsNum=neuronsInLayers.get(layerIdx);
        NeuronProperties outProperties=new NeuronProperties();
        outProperties.setProperty("transferFunction", TransferFunctionType.LINEAR);

        layer=LayerFactory.createLayer(neuronsNum,outProperties);
        this.addLayer(layer);
        ConnectionFactory.fullConnect(prevLayer,layer);

        NeuralNetworkFactory.setDefaultIO(this);

        this.setLearningRule(new BackPropagation());
        this.randomizeWeights(new NguyenWidrowRandomizer(-0.7,0.7));
    }
}
