package geym.nn.bam;

import ch.qos.logback.core.joran.conditional.PropertyWrapperForScripts;
import org.neuroph.core.Layer;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.Neuron;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.events.NeuralNetworkEvent;
import org.neuroph.core.learning.LearningRule;
import org.neuroph.nnet.comp.neuron.InputOutputNeuron;
import org.neuroph.util.*;

import java.util.ArrayList;
import java.util.List;

/**
 * BAM网络——双向双层神经网络
 */
public class BAM extends NeuralNetwork<LearningRule> {
    public BAM(int inputNeuronsCount, int outputNeuronsCount){
        NeuronProperties neuronProperties=new NeuronProperties();
        neuronProperties.setProperty("neuronType", InputOutputNeuron.class);
        neuronProperties.setProperty("bias", new Double(0));
        neuronProperties.setProperty("transferFunction", TransferFunctionType.SGN);

        this.createNetWork(inputNeuronsCount, outputNeuronsCount, neuronProperties);
    }

    public void switchInputOutput(){
        List<Neuron> intList= this.getInputNeurons();
        List<Neuron> outList=this.getOutputNeurons();

        List<Neuron> tmpList=new ArrayList<Neuron>(intList);

        intList.clear();
        this.setInputNeurons(outList);

        outList.clear();
        this.setOutputNeurons(tmpList);

        List<Layer> layers=new ArrayList<Layer>(getLayers());
        Layer t=layers.get(0);
        layers.set(0, layers.get(1));
        layers.set(1, t);
    }

    public void  switchInputOutputData(DataSetRow row){
        double[] t= row.getInput();

        row.setInput(row.getDesiredOutput());
        row.setDesiredOutput(t);
    }

    public boolean propagateLayer(DataSetRow row){
        this.setInput(row.getInput());

        for (Layer layer: this.getLayers()){

            layer.calculate();
        }

        boolean stable=true;

        double[] output=this.getOutput();
        for (int i=0; i<output.length; i++){
            if (output[i] != row.getDesiredOutput()[i]) {
                row.getDesiredOutput()[i] = output[i];
                stable = false;
            }
        }

        return stable;
    }

    public void calculate(DataSetRow row) {
        boolean stable1=true;
        boolean stable2=true;

        do {
            stable1=propagateLayer(row);
            switchInputOutput();
            switchInputOutputData(row);
            stable2=propagateLayer(row);
            switchInputOutput();
            switchInputOutputData(row);
        }while (!stable1 || !stable2);

        fireNetworkEvent(new NeuralNetworkEvent(this, NeuralNetworkEvent.Type.CALCULATED));
    }

    private void createNetWork(int inputNeuronsCount, int outputNeuronsCount, NeuronProperties neuronProperties) {
        this.setNetworkType(NeuralNetworkType.BAM);

        Layer inputLayer= LayerFactory.createLayer(inputNeuronsCount, neuronProperties);
        this.addLayer(inputLayer);

        Layer outputLayer=LayerFactory.createLayer(outputNeuronsCount, neuronProperties);
        this.addLayer(outputLayer);

        ConnectionFactory.fullConnect(inputLayer, outputLayer,0);
        ConnectionFactory.fullConnect(outputLayer, inputLayer, 0);
        NeuralNetworkFactory.setDefaultIO(this);
        this.setLearningRule(new BAMLearningRule());
    }
}
