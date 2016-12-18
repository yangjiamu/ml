import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by yangwenjie on 16/12/18.
 */
public class PerceptronClassifierDualForm {
    private double[] a;
    private double[] w;
    private double b;
    private double[][] gram;
    private List<TrainData> trainDatas;
    private double learningRate;
    public PerceptronClassifierDualForm(List<TrainData> trainDatas, double learningRate){
        this.trainDatas = trainDatas;
        this.learningRate = learningRate;
        w = new double[trainDatas.get(0).getFeatureDimension()];
        a = new double[trainDatas.size()];
        gram = new double[trainDatas.size()][trainDatas.size()];
        Arrays.fill(a, 0);
        b = 0;
    }

    public void learn(){
        generateGramMatrix();
        while (true){
            int numOfMisclassifiedTrainData = 0;
            for (int i = 0; i < trainDatas.size(); i++) {
                if(trainDatas.get(i).y * classify(i) <= 0){//misclassified
                    ++numOfMisclassifiedTrainData;
                    updateAAndB(i);
                }
            }
            if(numOfMisclassifiedTrainData == 0){
                break;
            }
        }
        //compute w vector
        for (int i = 0; i < w.length; i++) {
            double acc = 0;
            for (int j = 0; j < trainDatas.size(); j++) {
                acc += a[j] * trainDatas.get(j).y * trainDatas.get(j).getFeatureValueInDimension(i);
            }
            w[i] = acc;
        }
    }

    public int classify(Feature feature){
        if(dotProduct(w, feature.x) + b <= 0){
            return -1;
        }
        else {
            return 1;
        }
    }

    private int classify(int indexOfTrainData){
        double acc = 0;
        for (int j = 0; j < trainDatas.size(); j++) {
            acc += a[j] * trainDatas.get(j).y * gram[j][indexOfTrainData];
        }
        if(acc + b <= 0){
            return -1;
        }
        else {
            return 1;
        }
    }

    private void updateAAndB(int indexOfTrainData){
        a[indexOfTrainData] += learningRate;
        b += learningRate * trainDatas.get(indexOfTrainData).y;
    }

    public void generateGramMatrix(){
        for (int i = 0; i < gram.length; i++) {
            for (int j = 0; j < gram.length; j++) {
                gram[i][j] = computeGramMatrixValue(i, j);
            }
        }
    }

    public double[] getW(){
        return w;
    }
    public double getB(){
        return b;
    }

    private double computeGramMatrixValue(int x, int y){
        return dotProduct(trainDatas.get(x).getFeature(), trainDatas.get(y).getFeature());
    }

    private double dotProduct(double[] a, double[] b){
        double ret = 0;
        for (int i = 0; i < a.length; i++) {
            ret += a[i] * b[i];
        }
        return ret;
    }

    static class Feature{
        public double[] x;
        public Feature(double[] x){
            this.x = x;
        }

        public int getDimension(){
            return x.length;
        }

        public double getFeatureValueInDimension(int index){
            return x[index];
        }
    }

    static class TrainData{
        public Feature feature;
        public int y;

        public TrainData(Feature feature, int y){
            this.feature = feature;
            this.y = y;
        }

        public double[] getFeature(){
            return feature.x;
        }

        public int getFeatureDimension(){
            return feature.getDimension();
        }

        public double getFeatureValueInDimension(int index){
            return feature.getFeatureValueInDimension(index);
        }
    }

    public static void main(String[] args) {
        List<TrainData> trainDatas = new ArrayList<TrainData>();
        trainDatas.add(new TrainData(new Feature(new double[]{3, 3}), 1));
        trainDatas.add(new TrainData(new Feature(new double[]{1, 1}), -1));
        trainDatas.add(new TrainData(new Feature(new double[]{4, 3}), 1));
        PerceptronClassifierDualForm perceptronClassifierDualForm = new PerceptronClassifierDualForm(trainDatas, 1);
        perceptronClassifierDualForm.learn();
        double[] w = perceptronClassifierDualForm.getW();
        for (double v : w) {
            System.out.print(v + " ");
        }
        System.out.println();
        System.out.println(perceptronClassifierDualForm.getB());
        System.out.println(perceptronClassifierDualForm.classify(new Feature(new double[]{3, 3})));
        System.out.println(perceptronClassifierDualForm.classify(new Feature(new double[]{1, 1})));
        System.out.println(perceptronClassifierDualForm.classify(new Feature(new double[]{4, 3})));
        System.out.println(perceptronClassifierDualForm.classify(new Feature(new double[]{1, 2})));
    }
}
