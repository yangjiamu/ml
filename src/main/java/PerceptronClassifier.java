import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by yangwenjie on 16/12/17.
 */
public class PerceptronClassifier {
    private double[] w;
    private double b;
    private double learningRate;
    private List<TrainData> trainDatas;

    /*public PerceptronClassifier(List<TrainData> trainDatas){
        this(trainDatas, 1);
    }*/

    public PerceptronClassifier(List<TrainData> trainDatas, double learningRate){
        this.trainDatas = trainDatas;
        this.learningRate = learningRate;
        this.w = new double[trainDatas.get(0).getFeatureDimension()];
        Arrays.fill(w, 0);
        b = 0;
    }

    public void learn(){
        while (true){
            int numOfMisclassifiedTrainData = 0;
            for (int i = 0; i < trainDatas.size(); i++) {
                TrainData trainData = trainDatas.get(i);
                if(trainData.y * classify(trainData.feature) <= 0){
                    ++numOfMisclassifiedTrainData;
                    updateWAndB(trainData);
                }
            }
            if(numOfMisclassifiedTrainData == 0){
                break;
            }
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

    public double[] getW(){
        return w;
    }

    public double getB(){
        return b;
    }

    private void updateWAndB(TrainData trainData){
        for (int i = 0; i < w.length; i++) {
            w[i] += learningRate * trainData.y * trainData.getFeatureValueInDimension(i);
        }
        b += learningRate * trainData.y;
    }

    private double dotProduct(double[] w, double[] x){
        double ret = 0;
        for (int i = 0; i < w.length; i++) {
            ret += w[i]*x[i];
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

        PerceptronClassifier perceptronClassifier = new PerceptronClassifier(trainDatas, 1);
        perceptronClassifier.learn();
        for (double v : perceptronClassifier.getW()) {
            System.out.print(v + " ");
        }
        System.out.println();
        System.out.println(perceptronClassifier.getB());

        System.out.println(perceptronClassifier.classify(new Feature(new double[]{1, 2})));
    }
}
