package geym.nn.bam;

import org.neuroph.core.Neuron;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;

public class BAMLearningRule extends org.neuroph.core.learning.LearningRule {
    public void learn(DataSet trainingSet) {
        int M=trainingSet.size();

        for (int i=0; i<M; i++){
            DataSetRow dataSetRow=trainingSet.getRowAt(i);

            learnRow(dataSetRow);
        }
    }

    /**
     * 实现矩阵的外积算法 INPUT X (OUTPUT)T
     * @param row
     */
    public void learnRow(DataSetRow row){
        for (int i = 0; i < row.getInput().length; i++) {
            for (int j = 0; j < row.getDesiredOutput().length; j++) {
                Neuron ini=neuralNetwork.getLayerAt(0).getNeuronAt(i);  //提取第一层的第i个神经节点
                Neuron outj=neuralNetwork.getLayerAt(1).getNeuronAt(j); //提取第二层的第j个神经节点

                outj.getConnectionFrom(ini).getWeight().value +=
                        row.getInput()[i] * row.getDesiredOutput()[j];
                ini.getConnectionFrom(outj).getWeight().value=
                        outj.getConnectionFrom(ini).getWeight().value;
            }
        }
    }
}
