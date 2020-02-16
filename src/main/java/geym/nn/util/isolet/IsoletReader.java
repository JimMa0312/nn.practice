package geym.nn.util.isolet;

import geym.nn.util.Utils;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;

import java.io.*;

/**
 * 用于处理ISOLE数据文件的工具类
 */
public class IsoletReader {
    public static void unzipFileIfNetExist(String trainFile) throws IOException{
        File fImgFile=new File(trainFile);
        if(!fImgFile.exists()){
            Utils.decompress(trainFile+".Z");
        }
    }

    public static DataSet trainingData(String trainFile) throws FileNotFoundException,IOException{
        unzipFileIfNetExist(trainFile);

        DataSet ds=new DataSet(617,26);
        File fTrainFile=new File(trainFile);
        BufferedReader br=new BufferedReader(new FileReader(fTrainFile));
        String line=null;

        while (null != (line=br.readLine())){
            String[] strNumber=line.substring(0,line.length()-1).split(",");//最后一位是点
            double[] input=new double[strNumber.length-1];
            double[] otuput=new double[26];

            for (int i=0; i<strNumber.length-1; i++){
                input[i]=Double.parseDouble(strNumber[i].trim());
            }

            otuput[Integer.parseInt(strNumber[strNumber.length-1].trim())-1]=1;
            ds.add(new DataSetRow(input,otuput));
        }

        br.close();

        Utils.deleteFile(trainFile);

        return ds;
    }
}
