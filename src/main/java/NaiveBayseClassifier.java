import java.util.*;

/**
 * Created by yangwenjie on 16/12/18.
 */
public class NaiveBayseClassifier {
    private List<TrainData> trainDatas;
    private Map<String, Integer> categoryCountMap;
    private Map<String, Double> priorProbabilityMap;
    private Map<String, List<List<String>>> categoryToFeatures;
    private Map<String, Map<String, Double>> probabilityMatrix;

    public NaiveBayseClassifier(List<TrainData> trainDatas){
        this.trainDatas = trainDatas;
        categoryCountMap = new HashMap<String, Integer>();
        priorProbabilityMap = new HashMap<String, Double>();
        categoryToFeatures = new HashMap<String, List<List<String>>>();
        probabilityMatrix = new HashMap<String, Map<String, Double>>();
    }

    public void learn(){
        for (TrainData trainData : trainDatas) {
            if(categoryCountMap.containsKey(trainData.category)){
                categoryCountMap.put(trainData.category, categoryCountMap.get(trainData.category) + 1);
            }
            else {
                categoryCountMap.put(trainData.category, 1);
            }
            if(categoryToFeatures.containsKey(trainData.category)){
                categoryToFeatures.get(trainData.category).add(trainData.feature);
            }
            else {
                List<List<String>> featureList = new ArrayList<List<String>>();
                featureList.add(trainData.feature);
                categoryToFeatures.put(trainData.category, featureList);
            }
        }
        //compute prior probability
        for (Map.Entry<String, Integer> entry : categoryCountMap.entrySet()) {
            priorProbabilityMap.put(entry.getKey(), (double)entry.getValue()/trainDatas.size());
            //priorProbabilityMap.put(entry.getKey(), (double)(entry.getValue() + 1) / (trainDatas.size() + categoryCountMap.size()));//拉普拉斯平滑
        }
        //compute probability matrix
        for (Map.Entry<String, List<List<String>>> entry : categoryToFeatures.entrySet()) {
            String category = entry.getKey();
            List<List<String>> featureList = entry.getValue();
            Map<String, Integer> curCateWordsCount = new HashMap<String, Integer>();
            for (List<String> feature : featureList) {
                for (String word : feature) {
                    if(curCateWordsCount.containsKey(word)){
                        curCateWordsCount.put(word, curCateWordsCount.get(word) + 1);
                    }
                    else {
                        curCateWordsCount.put(word, 1);
                    }
                }
            }
            Map<String, Double> curCateWordsProbabilityMap = new HashMap<String, Double>();
            for (Map.Entry<String, Integer> stringIntegerEntry : curCateWordsCount.entrySet()) {
                curCateWordsProbabilityMap.put(stringIntegerEntry.getKey(), (double)stringIntegerEntry.getValue()/categoryCountMap.get(category));
            }
            probabilityMatrix.put(category, curCateWordsProbabilityMap);
        }
    }

    public String classify(List<String> feature){
        double maxProbability = Double.MIN_VALUE;
        String category = null;
        for (Map.Entry<String, Map<String, Double>> entry : probabilityMatrix.entrySet()) {
            double curCategoryProbability = 1;
            Map<String, Double> curCateWordsProbabilityMap = entry.getValue();
            for (String word : feature) {
                if(curCateWordsProbabilityMap.containsKey(word)) {
                    curCategoryProbability *= curCateWordsProbabilityMap.get(word);
                }
                else {
                    //拉普拉斯平滑
                }
            }
            if(Double.compare(curCategoryProbability, maxProbability) > 0){
                category = entry.getKey();
            }
        }
        return category;
    }

    static class TrainData{
        public List<String> feature;
        public String category;

        public TrainData(List<String> feature, String category){
            this.feature = feature;
            this.category = category;
        }
    }

    public static void main(String[] args) {
        simpleTest();
    }

    public static void testSouGouData(){
        List<TrainData> trainDatas = loadDataFromFile();
        NaiveBayseClassifier naiveBayseClassifier = new NaiveBayseClassifier(trainDatas);
        naiveBayseClassifier.learn();
    }

    public static List<TrainData> loadDataFromFile(){
        List<TrainData> trainDatas = new ArrayList<TrainData>();
        String path = "";

        return trainDatas;
    }

    public static void simpleTest(){
        List<TrainData> trainDatas = new ArrayList<TrainData>();
        trainDatas.add(new TrainData(Arrays.asList(new String[]{"1", "S"}), "-1"));
        trainDatas.add(new TrainData(Arrays.asList(new String[]{"1", "M"}), "-1"));
        trainDatas.add(new TrainData(Arrays.asList(new String[]{"1", "M"}), "1"));
        trainDatas.add(new TrainData(Arrays.asList(new String[]{"1", "S"}), "1"));
        trainDatas.add(new TrainData(Arrays.asList(new String[]{"1", "S"}), "-1"));
        trainDatas.add(new TrainData(Arrays.asList(new String[]{"2", "S"}), "-1"));
        trainDatas.add(new TrainData(Arrays.asList(new String[]{"2", "M"}), "-1"));
        trainDatas.add(new TrainData(Arrays.asList(new String[]{"2", "M"}), "1"));
        trainDatas.add(new TrainData(Arrays.asList(new String[]{"2", "L"}), "1"));
        trainDatas.add(new TrainData(Arrays.asList(new String[]{"2", "L"}), "1"));
        trainDatas.add(new TrainData(Arrays.asList(new String[]{"3", "L"}), "1"));
        trainDatas.add(new TrainData(Arrays.asList(new String[]{"3", "M"}), "1"));
        trainDatas.add(new TrainData(Arrays.asList(new String[]{"3", "M"}), "1"));
        trainDatas.add(new TrainData(Arrays.asList(new String[]{"3", "L"}), "1"));
        trainDatas.add(new TrainData(Arrays.asList(new String[]{"3", "L"}), "-1"));
        NaiveBayseClassifier naiveBayseClassifier = new NaiveBayseClassifier(trainDatas);
        naiveBayseClassifier.learn();
        System.out.println(naiveBayseClassifier.classify(Arrays.asList(new String[]{"2", "S"})));
    }
}
