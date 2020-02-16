package geym.nn.util.mnist;

import geym.nn.util.Utils;
import org.neuroph.core.data.DataSet;

import javax.rmi.CORBA.Util;
import javax.swing.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class MnistReader {
    public static void unzipFileIfNetExist(String trainFile) throws IOException {
        File fImgFile=new File(trainFile);
        if(!fImgFile.exists()){
            Utils.decompress(trainFile+".gz");
        }
    }

    public static BufferedImage[] readImg(String imgFile, int indexBefore) throws IOException{
        unzipFileIfNetExist(imgFile);

        DataInputStream dis=new DataInputStream(new BufferedInputStream(new FileInputStream(new File(imgFile))));

        try{
            int magic=dis.readInt();

            if (magic != 0x00000803){
                throw new RuntimeException("net a mnist imagin file");
            }

            int numberOfImg=dis.readInt();
            if (indexBefore>numberOfImg){
                throw new RuntimeException("max number of image is "+numberOfImg);
            }
            int rows=dis.readInt();
            int cols=dis.readInt();

            BufferedImage[] re=new BufferedImage[indexBefore];
            byte[] bImg=new byte[rows*cols];
            for (int i=0; i<indexBefore; i++){
                dis.read(bImg);
                re[i] = createImgByBytes(bImg, rows, cols);
            }

            return re;
        }finally {
            dis.close();
        }
    }

    /**
     * 根据像素 进行Img着色
     * @param bImg
     * @param rows
     * @param cols
     * @return
     */
    public static BufferedImage createImgByBytes(byte[] bImg, int rows, int cols) {
        if(cols*rows!=bImg.length){
            throw new RuntimeException("image length error");
        }

        BufferedImage bi=new BufferedImage(rows, cols, BufferedImage.TYPE_INT_RGB);
        for (int p=0; p<bImg.length;p++){
            bImg[p]=(byte)(255-bImg[p]);
            bi.setRGB(p%rows, p/rows, (bImg[p]<<16|bImg[p]<<8|bImg[p]));
        }

        return bi;
    }

    /**
     * 根据 1和0进行着色
     * @param bImg
     * @param rows
     * @param cols
     * @return
     */
    public static BufferedImage craeteImgByBytes2(byte[] bImg, int rows, int cols){
        if(cols*rows!=bImg.length){
            throw new RuntimeException("image length error");
        }

        BufferedImage bi=new BufferedImage(rows, cols, BufferedImage.TYPE_INT_RGB);
        for (int p=0; p<bImg.length;p++){
            if(bImg[p]==1){
                bImg[p]=(byte)0xFF;
            }
            bi.setRGB(p%rows, p/rows, (bImg[p]<<16|bImg[p]<<8|bImg[p]));
        }

        return bi;
    }

    public static List<byte[]> readImgAsBytes(String imgFile, int indexBefore) throws IOException{
        unzipFileIfNetExist(imgFile);
        DataInputStream dis=new DataInputStream(new BufferedInputStream(new FileInputStream(new File(imgFile))));
        try{
            int magic=dis.readInt();
            if(magic!=0x00000803){
                throw new RuntimeException("not a mnist imagin file");
            }
            int numberOfImg=dis.readInt();
            if (indexBefore>numberOfImg){
                throw new RuntimeException("max number of image is "+numberOfImg);
            }

            int rows=dis.readInt();
            int cols=dis.readInt();

            List<byte[]> re=new ArrayList<byte[]>(indexBefore);
            for (int i = 0; i < indexBefore; i++) {
                byte[] bImg=new byte[rows*cols];
                dis.read(bImg);
                for (int k=0; k<bImg.length; k++){
                    if ((bImg[k]&0xFF)<128){
                        bImg[k]=0;
                    }
                    else{
                        bImg[k]=1;
                    }
                }
                re.add(bImg);
            }
            return re;
        }finally {
            dis.close();
        }
    }

    /**
     * 通过Label文件读取图像映射
     * @param imgLabelFile
     * @param indexBefore
     * @return
     * @throws IOException
     */
    public static byte[] readImgLabel(String imgLabelFile, int indexBefore) throws IOException{
        unzipFileIfNetExist(imgLabelFile);
        DataInputStream dis=new DataInputStream(new BufferedInputStream(new FileInputStream(new File(imgLabelFile))));

        try {
            int magic=dis.readInt();
            if(magic != 0x00000801){
                throw new RuntimeException("not a mnist imagin label file");
            }
            int numberOfImg=dis.readInt();
            if (indexBefore>numberOfImg){
                throw new RuntimeException("max number of image is " + numberOfImg);
            }

            byte[] lables=new byte[indexBefore];
            dis.read(lables);
            return lables;
        }finally {
            dis.close();
        }
    }

    public static DataSet trainingData(String imgFile, String imgLabelFile, int indexBefore)throws IOException{
        List<byte[]> inputs=readImgAsBytes(imgFile, indexBefore);
        byte[] labels=readImgLabel(imgLabelFile, indexBefore);
        DataSet ds=new DataSet(inputs.get(0).length,10);
        for (int i = 0; i < inputs.size(); i++) {
            ds.add(encodeInput(inputs.get(i)),encodeOutput(labels[i]));
        }

        Utils.deleteFile(imgFile);
        Utils.deleteFile(imgLabelFile);
        return ds;
    }

    private static double[] encodeInput(byte[] bs){
        double[] re=new double[bs.length];
        for (int i=0; i<bs.length;i++){
            re[i]=bs[i];
        }
        return re;
    }

    private static double[] encodeInput2(byte[] bs){
        double[] re=new double[bs.length];
        for (int i=0; i<bs.length;i++){
            re[i]=bs[i]&0xff;
        }
        return re;
    }

    private static double[] encodeOutput(byte b){
        return NumberOutput.numbers.get(b);
    }
}
