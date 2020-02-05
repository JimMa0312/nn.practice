package geym.nn.perceptron.rule;

import org.neuroph.core.Connection;
import org.neuroph.core.Neuron;
import org.neuroph.core.Weight;
import org.neuroph.core.learning.SupervisedLearning;

import java.io.Serializable;

public class PerceptronLearningRule extends SupervisedLearning implements Serializable {

    private static final long serialVersionUID=1L;

    public PerceptronLearningRule() {
    }

    /**
     * 本方法用来实现根据错误向量
     * @param outputError 错误向量势根据实际输出与预期值的差得到的
     */
    protected void calculateWeightChanges(double[] outputError) {
        int i=0;
        for (Neuron neuron: neuralNetwork.getOutputNeurons()){
            double neuronError=outputError[i];
            for (Connection connection :
                    neuron.getInputConnections()) {
                double input = connection.getInput();
                double weightChange = neuronError * input;

                Weight weight = connection.getWeight();
                weight.weightChange = weightChange;
                weight.value += weightChange;
            }

            i++;
        }
    }
}
