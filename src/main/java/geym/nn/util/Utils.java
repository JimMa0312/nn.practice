package geym.nn.util;

import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream;
import org.apache.commons.compress.compressors.z.ZCompressorInputStream;

import java.io.*;

public class Utils {

    private static final int BUFFER=512;
    /**
     * 将活跃度最高的所在位置激活（索引位置被置为1）
     * @param d
     * @return
     */
    public static double[] competiton(double[] d){
        double[] output=d;
        double[] re=new double[output.length];
        int maxIndex=0;
        double maxValue=Double.MIN_VALUE;

        for (int i = 0; i < output.length; i++) {
            if (output[i]>maxValue){
                maxIndex=i;
                maxValue=output[i];
            }
        }

        for (int i = 0; i < re.length; i++) {
            if (i==maxIndex){
                re[i]=1;
            }else{
                re[i]=0;
            }
        }

        return re;
    }

    public static void decompress(String gzFile)throws IOException{
        int i=gzFile.lastIndexOf('.');
        String outputFileName=gzFile.substring(0,i);

        if(gzFile.endsWith("Z")){
            decompressZ(new BufferedInputStream(new FileInputStream(gzFile)),
                    new BufferedOutputStream(new FileOutputStream(outputFileName)));
        }else if(gzFile.endsWith("gz")){
            decompressGZ(new BufferedInputStream(new FileInputStream(gzFile)),
                    new BufferedOutputStream(new FileOutputStream(outputFileName)));
        }else{

        }
    }

    public static void deleteFile(String fileName) throws IOException{
        File file=new File(fileName);

        if (file.exists()){
            if(file.isFile()){
                file.delete();
            }
        }
    }

    /**
     * 使用Apache的工具 对Z,gz文件解压
     * @param is
     * @param os
     * @throws IOException
     */
    private static void decompressZ(InputStream is, OutputStream os) throws IOException{
        ZCompressorInputStream zin=new ZCompressorInputStream(is);

        try {

            int count;
            final byte data[]=new byte[BUFFER];

            while (-1 != (count = zin.read(data,0,BUFFER))){
                os.write(data,0,count);
            }
        }finally {
            os.close();
            zin.close();
            is.close();
        }
    }

    private static void decompressGZ(InputStream is, OutputStream os) throws IOException{
        GzipCompressorInputStream gzin=new GzipCompressorInputStream(is);

        try{
            int count;
            final byte data[]=new byte[BUFFER];

            while (-1 !=(count=gzin.read(data,0,BUFFER))){
                os.write(data,0,count);
            }
        }finally {
            os.close();
            gzin.close();
            is.close();
        }
    }
}
