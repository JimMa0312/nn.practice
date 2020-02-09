package geym.nn.mlperceptron;

import org.neuroph.core.Layer;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.transfer.Linear;
import org.neuroph.nnet.comp.neuron.BiasNeuron;
import org.neuroph.nnet.comp.neuron.InputNeuron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.*;
import org.neuroph.util.random.NguyenWidrowRandomizer;

import java.util.ArrayList;
import java.util.List;

public class MlPerceptron extends NeuralNetwork<BackPropagation> {
    private static final long serialVersionUID=2L;

    public MlPerceptron(List<Integer> neuronsInLayers){
        this(neuronsInLayers,TransferFunctionType.SIGMOID);
    }

    public MlPerceptron(int... neuronsInLayers){
        this(TransferFunctionType.SIGMOID, neuronsInLayers);
    }

    public MlPerceptron(TransferFunctionType transferFunctionType, int... neuronsInLayers){
        NeuronProperties neuronProperties=new NeuronProperties();

        neuronProperties.setProperty("transferFunction",transferFunctionType);
        List<Integer> neuronsInLayerVector=new ArrayList<Integer>();
        for (int neuronsInLayer : neuronsInLayers) {
            neuronsInLayerVector.add(neuronsInLayer);
        }

        this.createNetWork(neuronsInLayerVector,neuronProperties);
    }

    public MlPerceptron(List<Integer> neuronsInLayers, TransferFunctionType transferFunctionType){
        NeuronProperties neuronProperties=new NeuronProperties();
        neuronProperties.setProperty("transferFunction", transferFunctionType);
        this.createNetWork(neuronsInLayers,neuronProperties);
    }

    public MlPerceptron(List<Integer> neuronsInLayers, NeuronProperties neuronProperties){
        this.createNetWork(neuronsInLayers,neuronProperties);
    }

    protected void createNetWork(List<Integer> neuronsInLayers, NeuronProperties neuronProperties) {
        this.setNetworkType(NeuralNetworkType.MULTI_LAYER_PERCEPTRON);

        NeuronProperties inputNeuronProperties=new NeuronProperties(InputNeuron.class, Linear.class);
        Layer layer= LayerFactory.createLayer(neuronsInLayers.get(0),inputNeuronProperties);
        layer.addNeuron(new BiasNeuron());
        this.addLayer(layer);

        Layer preLayer=layer;
        for (int layerIdx = 1; layerIdx < neuronsInLayers.size(); layerIdx++) {
            Integer neuronsNum=neuronsInLayers.get(layerIdx);
            layer=LayerFactory.createLayer(neuronsNum,neuronProperties);
            this.addLayer(layer);

            if(layerIdx!=neuronsInLayers.size()-1){
                layer.addNeuron(new BiasNeuron());
            }

            ConnectionFactory.fullConnect(preLayer,layer);

            preLayer=layer;
        }

        NeuralNetworkFactory.setDefaultIO(this);
        this.setLearningRule(new BackPropagation());
        this.randomizeWeights(new NguyenWidrowRandomizer(-0.7,0.7));
    }


}
