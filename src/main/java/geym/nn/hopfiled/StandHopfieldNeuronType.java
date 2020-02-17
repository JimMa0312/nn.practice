package geym.nn.hopfiled;

import org.neuroph.nnet.comp.neuron.InputOutputNeuron;
import org.neuroph.util.NeuronProperties;
import org.neuroph.util.TransferFunctionType;

public class StandHopfieldNeuronType {
    public static NeuronProperties getHopfieldNeuronType(){
        NeuronProperties neuronProperties=new NeuronProperties();
        neuronProperties.setProperty("neuronType", InputOutputNeuron.class);
        neuronProperties.setProperty("bias",new Double(0));
        neuronProperties.setProperty("transferFunction", TransferFunctionType.SGN);
        return neuronProperties;
    }
}
