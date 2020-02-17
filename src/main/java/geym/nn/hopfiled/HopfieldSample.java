package geym.nn.hopfiled;

import geym.nn.adaline.demo.AdalineDemo;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.Hopfield;

import java.util.Arrays;

/**
 * 已记忆的图像进行还原
 *
 * BAD_DIGITS[0] 修复失败，正在检查原因
 */
public class HopfieldSample {

    public static String[][] DIGITS = {
            { " OOO ",
                    "O   O",
                    "O   O",
                    "O   O",
                    "O   O",
                    "O   O",
                    " OOO "  },

            { "  O  ",
                    " OO  ",
                    "O O  ",
                    "  O  ",
                    "  O  ",
                    "  O  ",
                    "  O  "  }
    };

    public static String[][] BAD_DIGITS = {
            { " OOO ",
                    "O   O",
                    "O  OO",
                    "O    ",
                    "OOOOO",
                    "OOOOO",
                    " OOO "  },
            { " OOO ",
                    "O   O",
                    "O    ",
                    "    O",
                    "O O O",
                    "OOO O",
                    " OOO "  },

            { "  O  ",
                    " OO  ",
                    "O O  ",
                    "    O",
                    "     ",
                    "     ",
                    "     "  }
    };

    public static void main(String[] args) {
        DataSet traningSet=new DataSet(35);
        traningSet.add(AdalineDemo.createTrainDataRow(DIGITS[0],0));
        traningSet.add(AdalineDemo.createTrainDataRow(DIGITS[1],0));

        Hopfield myHopfield=new Hopfield(35);
        myHopfield.setLearningRule(new StandHopfieldLearning());
        myHopfield.learn(traningSet);

        System.out.println("Testing network");
        for (int i=0;i<BAD_DIGITS.length;i++){
            recalDigit(myHopfield, BAD_DIGITS[i]);
        }
    }

    private static void recalDigit(Hopfield myHopfield, String[] bad_digit){
        DataSetRow h= AdalineDemo.createTrainDataRow(bad_digit,0);
        myHopfield.setInput(h.getInput());
        double[] networkOutput=null;
        double[] preNetworkOutput=null;

        while (true){
            myHopfield.calculate();
            networkOutput=myHopfield.getOutput();

            if(preNetworkOutput==null){
                preNetworkOutput=networkOutput;
                continue;
            }

            if (Arrays.equals(networkOutput, preNetworkOutput)){
                break;
            }
            preNetworkOutput=networkOutput;
        }

        System.out.println("Input: ");
        printDigit(h.getInput());
        System.out.println(" Output ====>" );
        printDigit(networkOutput);
    }

    public static void printDigit(double[] networkOutput){
        for (int i = 0; i < networkOutput.length; i++) {
            if (networkOutput[i]>0){
                System.out.print("O");
            }else{
                System.out.print(" ");
            }

            if (((i+1)%5)==0){
                System.out.println();
            }
        }
    }
}
