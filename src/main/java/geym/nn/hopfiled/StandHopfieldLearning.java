package geym.nn.hopfiled;

import org.neuroph.core.Connection;
import org.neuroph.core.Layer;
import org.neuroph.core.Neuron;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.learning.LearningRule;

/**
 * Hopfield神经网络的标准学习算法
 */
public class StandHopfieldLearning extends LearningRule {

    private static final long serialVersionUID=1L;

    public StandHopfieldLearning(){super();}

    /**
     * 学习算法
     *  外积和法
     *  非监督式学习
     *
     *  神经元i和神经元j的权值=每个训练样本上i分量与j分量乘积之和(当i==j时，积为0)
     *
     *  权值矩阵为对称矩阵
     *
     * @param trainingSet
     */
    public void learn(DataSet trainingSet) {
        int M=trainingSet.size();
        int N=neuralNetwork.getLayerAt(0).getNeuronsCount();

        Layer hopfieldLayer=neuralNetwork.getLayerAt(0);

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if(j==i){
                    continue;
                }

                Neuron ni=hopfieldLayer.getNeuronAt(i);
                Neuron nj=hopfieldLayer.getNeuronAt(j);

                Connection cij=nj.getConnectionFrom(ni);
                Connection cji=ni.getConnectionFrom(nj);

                double wij=0;

                for (int k=0; k<M;k++){
                    DataSetRow row=trainingSet.getRowAt(k);
                    double[] inputs=row.getInput();
                    wij+=inputs[i]*inputs[j];
                }

                cij.getWeight().setValue(wij);
                cji.getWeight().setValue(wij);
            }//j
        }//i
    }
}
