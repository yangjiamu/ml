import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by yangwenjie on 16/12/17.
 */
public class LogisticRegression {

    private List<TrainSample> samples;
    private double learningRate;
    private int iterations;
    private double[] w;
    public LogisticRegression(List<TrainSample> samples, double learningRate, int iterations){
        this.samples = samples;
        this.learningRate = learningRate;
        this.iterations = iterations;
        w = new double[samples.get(0).getFeatureDimension()];
        Arrays.fill(w, 0);
    }

    public void train(char c){
        switch (c){
            case 'S':
                stochasticGradientDescent();
                break;
            case 'B':
                batchGradientDescent();
                break;
        }
    }

    public void stochasticGradientDescent(){
        while (iterations-- > 0){
            for (int i = 0; i < samples.size(); i++) {
                double predicted = classify(samples.get(i).feature);
                double error = samples.get(i).label - predicted;
                for (int j = 0; j < w.length; j++) {
                    w[j] += learningRate * error * samples.get(i).getFeatureValueInIndex(j);
                }
            }
        }
    }

    public void batchGradientDescent(){
        while (iterations-- > 0){
            double[] errors = new double[samples.size()];
            for (int i = 0; i < samples.size(); i++) {
                double predicted = classify(samples.get(i).feature);
                errors[i] = samples.get(i).label - predicted;
            }
            for (int i = 0; i < w.length; i++) {
                double sum = 0;
                for (int j = 0; j < errors.length; j++) {
                    sum += errors[j] * samples.get(j).getFeatureValueInIndex(i);
                }
                w[i] += learningRate * sum;
            }
        }
    }

    public double classify(double[] feature){
        double z = 0;
        for (int i = 0; i < feature.length; i++) {
            z += w[i] * feature[i];
        }
        return sigmod(z);
    }

    public double sigmod(double z){
        return 1.0 / (1.0 + Math.exp(-z));
    }

    static class TrainSample{
        public double[] feature;
        public int label;

        public TrainSample(double[] feature, int label){
            this.feature = feature;
            this.label = label;
        }

        public int getFeatureDimension(){
            return feature.length;
        }

        public double getFeatureValueInIndex(int index){
            return feature[index];
        }
    }

    public static void main(String[] args) throws IOException {
        List<TrainSample> samples = loadData();
        LogisticRegression lr = new LogisticRegression(samples, 0.001, 90000);
        lr.train('S');
        List<Double> sResult = new ArrayList<Double>();
        for (TrainSample sample : samples) {
            sResult.add(lr.classify(sample.feature));
        }
        lr = new LogisticRegression(samples, 0.001, 900000);
        lr.train('B');
        for (int i = 0; i < samples.size(); i++) {
            System.out.println("Stochastic " + sResult.get(i) + " Batch " + lr.classify(samples.get(i).feature) + " label " + samples.get(i).label);
        }
    }

    public static List<TrainSample> loadData() throws IOException {
        String path = LogisticRegression.class.getResource("/LR.data1").getPath();
        //String path = "resources/LR.data";
        BufferedReader bufferedReader = new BufferedReader(new FileReader(new File(path)));
        List<TrainSample> samples = new ArrayList<TrainSample>();
        String line = null;
        while ((line = bufferedReader.readLine()) != null){
            if(line.startsWith("#")){
                continue;
            }
            String[] split = line.split("\\s+");
            double[] feature = new double[split.length-1];
            for (int i = 0; i < split.length - 1; i++) {
                feature[i] = Double.parseDouble(split[i]);
            }
            samples.add(new TrainSample(feature, Integer.parseInt(split[split.length-1])));
        }
        return samples;
    }
}
