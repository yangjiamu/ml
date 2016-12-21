import org.junit.Test;

import java.util.*;

/**
 * Created by yangwenjie on 16/12/19.
 */
public class DecisionTree {
    private List<TrainData> trainDatas;
    private List<String> attributes;
    public DecisionTree(List<TrainData> trainDatas, List<String> attributes){
        this.trainDatas = trainDatas;
        this.attributes = attributes;
    }

    public TreeNode learn(){
        return generateTree(trainDatas, attributes);
    }

    public String classify(){
        return null;
    }

    public TreeNode generateTree(List<TrainData> datas, List<String> attributes){
        TreeNode treeNode = new TreeNode();
        if(isAllSameClass(datas)){
            treeNode.labelOrSplitAttrName = datas.get(0).label;
            return treeNode;
        }
        int bestAttributeIndex = CriteriaID3.selectAttribute(datas);
        treeNode.labelOrSplitAttrName = attributes.get(bestAttributeIndex);
        treeNode.setRules(getAttrValueSet(datas, bestAttributeIndex));
        Map<String, List<TrainData>> attributeToDatas = CriteriaID3.splitByAttribute(datas, bestAttributeIndex);
        List<TreeNode> children = treeNode.getChildren();
        for (Map.Entry<String, List<TrainData>> entry : attributeToDatas.entrySet()) {
            children.add(generateTree(entry.getValue(), attributes));
        }
        return treeNode;
    }

    public boolean isAllSameClass(List<TrainData> datas){
        if(datas.isEmpty() || datas.size() == 1){
            return true;
        }
        int i;
        for (i = 1; i < datas.size(); i++) {
            if(!datas.get(i).label.equals(datas.get(i-1).label)){
                break;
            }
        }
        return i == datas.size();
    }

    public List<String> getAttrValueSet(List<TrainData> datas, int index){
        List<String> attributeValues = new ArrayList<String>();
        for (TrainData data : datas) {
            if(!attributeValues.contains(data.getFeatureValueByIndex(index))){
                attributeValues.add(data.getFeatureValueByIndex(index));
            }
        }
        return attributeValues;
    }

    public void print(TreeNode root){
        Queue<TreeNode> q = new LinkedList<TreeNode>();
        Queue<TreeNode> nextLevel = new LinkedList<TreeNode>();
        q.add(root);
        while (!q.isEmpty()){
            while (!q.isEmpty()){
                TreeNode node = q.remove();
                for (TreeNode child : node.getChildren()) {
                    nextLevel.add(child);
                }
                System.out.print(node.labelOrSplitAttrName + " ");
            }
            System.out.println();
            Queue<TreeNode> t = q;
            q = nextLevel;
            nextLevel = t;
        }
    }


    static class CriteriaID3{
        public static int selectAttribute(List<TrainData> datas){
            double entropy = calculateEntropy(datas);//not used
            int numOfAttribute = datas.get(0).getFeatureDimension();
            double minConditionEntropy = Double.MAX_VALUE;
            int bestAttributeIndex = Integer.MAX_VALUE;
            for (int i = 0; i < numOfAttribute; i++) {
                double conditionEntropy = 0;
                Map<String, List<TrainData>> attributeToDatas = splitByAttribute(datas, i);
                for (Map.Entry<String, List<TrainData>> entry : attributeToDatas.entrySet()) {
                    conditionEntropy += (1.0d * entry.getValue().size() / datas.size() * calculateEntropy(entry.getValue()));
                }
                if(conditionEntropy < minConditionEntropy){
                    minConditionEntropy = conditionEntropy;
                    bestAttributeIndex = i;
                }
            }
            return bestAttributeIndex;
        }

        public static double calculateEntropy(List<TrainData> datas){
            Map<String, Integer> labelCountMap = new HashMap<String, Integer>();
            for (TrainData data : datas) {
                int count = labelCountMap.containsKey(data.label) ? labelCountMap.get(data.label) + 1 : 1;
                labelCountMap.put(data.label, count);
            }
            double entropy = 0;
            for (Map.Entry<String, Integer> entry : labelCountMap.entrySet()) {
                double t = 1.0d * entry.getValue() / datas.size();
                entropy += (t * Math.log(t) / Math.log(2));
            }
            return -entropy;
        }

        public static Map<String, List<TrainData>> splitByAttribute(List<TrainData> datas, int attributeIndex){
            Map<String, List<TrainData>> attributeToDatasMap = new HashMap<String, List<TrainData>>();
            for (TrainData data : datas) {
                String attributeValue = data.getFeatureValueByIndex(attributeIndex);
                if(attributeToDatasMap.containsKey(attributeValue)){
                    attributeToDatasMap.get(attributeValue).add(data);
                }
                else {
                    List<TrainData> list = new ArrayList<TrainData>();
                    list.add(data);
                    attributeToDatasMap.put(attributeValue, list);
                }
            }
            return attributeToDatasMap;
        }
    }

    static class TrainData{
        public List<String> features;
        public String label;

        public TrainData(List<String> features, String label){
            this.features = features;
            this.label = label;
        }

        public String getFeatureValueByIndex(int index){
            return features.get(index);
        }

        public int getFeatureDimension(){
            return features.size();
        }
    }

    static class TreeNode{
        public String labelOrSplitAttrName;
        public List<String> rules;
        public List<TreeNode> children;

        public TreeNode(){
            this.rules = new ArrayList<String>();
            this.children = new ArrayList<TreeNode>();
        }

        public void setRules(List<String> rules){
            this.rules = rules;
        }
        public List<String> getRules(){
            return rules;
        }

        public void setChildren(List<TreeNode> children){
            this.children = children;
        }
        public List<TreeNode> getChildren(){
            return this.children;
        }
    }

    public static void main(String[] args) {
        testGenerateTree();
    }
    public static void testGenerateTree(){
        List<TrainData> datas = new ArrayList<TrainData>();
        datas.add(new TrainData(Arrays.asList(new String[]{"young", "high", "no", "fair"}), "no"));
        datas.add(new TrainData(Arrays.asList(new String[]{"young", "high", "no", "excellent"}), "no"));
        datas.add(new TrainData(Arrays.asList(new String[]{"middle", "high", "no", "fair"}), "yes"));
        datas.add(new TrainData(Arrays.asList(new String[]{"senior", "medium", "no", "fair"}), "yes"));
        datas.add(new TrainData(Arrays.asList(new String[]{"senior", "low", "yes", "fair"}), "yes"));
        datas.add(new TrainData(Arrays.asList(new String[]{"senior", "low", "yes", "excellent"}), "no"));
        datas.add(new TrainData(Arrays.asList(new String[]{"middle", "low", "yes", "excellent"}), "yes"));
        datas.add(new TrainData(Arrays.asList(new String[]{"young", "medium", "no", "fair"}), "no"));
        datas.add(new TrainData(Arrays.asList(new String[]{"young", "low", "yes", "fair"}), "yes"));
        datas.add(new TrainData(Arrays.asList(new String[]{"senior", "medium", "yes", "fair"}), "yes"));
        datas.add(new TrainData(Arrays.asList(new String[]{"young", "medium", "yes", "excellent"}), "yes"));
        datas.add(new TrainData(Arrays.asList(new String[]{"middle", "medium", "no", "excellent"}), "yes"));
        datas.add(new TrainData(Arrays.asList(new String[]{"middle", "high", "yes", "fair"}), "yes"));
        datas.add(new TrainData(Arrays.asList(new String[]{"senior", "medium", "no", "excellent"}), "no"));

        List<String> att = Arrays.asList(new String[]{"age", "income", "student", "credit_rating", "buys_computer"});
        DecisionTree decisionTree = new DecisionTree(datas, att);
        TreeNode root = decisionTree.learn();
        decisionTree.print(root);
    }
}
