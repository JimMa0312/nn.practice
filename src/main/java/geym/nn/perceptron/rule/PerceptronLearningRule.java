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
     * @param outputError 错误向量势根据实际输出与预期值的差得到的 根据2.96源代码分析
     *                    该错误向量计算关系是e=a-t（e：误差，a：实际输出，t：期望）
     *                    理论上一般使用的公式是e=t-a
     *                    因此可推e=-(a-t)
     */
    protected void calculateWeightChanges(double[] outputError) {
        int i=0;
        for (Neuron neuron: neuralNetwork.getOutputNeurons()){
            double neuronError=outputError[i];
            for (Connection connection :
                    neuron.getInputConnections()) {
                double input = connection.getInput();
                double weightChange = -(neuronError * input);

                Weight weight = connection.getWeight();
                weight.weightChange = weightChange;
                weight.value += weightChange;
            }
            i++;
        }
    }
}
