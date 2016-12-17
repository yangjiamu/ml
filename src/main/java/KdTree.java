import org.junit.Test;

import java.util.*;

/**
 * Created by yangwenjie on 16/12/16.
 */
public class KdTree {

    public KdNode buildKdTree(List<KdNode> nodes){
        return buildKdTree(nodes, 0);
    }

    public KdNode buildKdTree(List<KdNode> nodes, int dimension){
        if(nodes == null || nodes.size()==0){
            return null;
        }
        sortByDimension(nodes, dimension);
        KdNode root = nodes.get(nodes.size()/2);
        root.dimension = dimension;
        List<KdNode> left = new ArrayList<KdNode>();
        List<KdNode> right = new ArrayList<KdNode>();
        for (KdNode node : nodes) {
            if(root != node){
                if(Double.compare(node.getFeatureInDimension(dimension), root.getFeatureInDimension(dimension)) <= 0){
                    left.add(node);
                }
                else {
                    right.add(node);
                }
            }
        }
        ++dimension;
        dimension %= root.data.length;
        root.left = buildKdTree(left, dimension);
        root.right = buildKdTree(right, dimension);
        if(root.left != null){
            root.left.parent = root;
        }
        if(root.right != null){
            root.right.parent = root;
        }
        return root;
    }

    public List<KdNode> searchKnn(KdNode root, KdNode query, int k){
        List<KdNode> knnNodes = new ArrayList<KdNode>(k);
        KdNode almostNearstNode = searchLeaf(root, query);//包含query节点的近似最近叶子节点
        while (almostNearstNode != null){
            double curDistance = almostNearstNode.computeDistance(query);
            almostNearstNode.distance = curDistance;
            mainMaxHeap(knnNodes, almostNearstNode, k);
            if(almostNearstNode.parent != null &&
                    curDistance > almostNearstNode.parent.computeDistance(query)){
                KdNode brother = almostNearstNode.getBrother();
                mainMaxHeap(knnNodes, brother, k);
            }
            almostNearstNode = almostNearstNode.parent;
        }
        return knnNodes;
    }

    private KdNode searchLeaf(KdNode root, KdNode query){
        int dimension = 0;
        KdNode p = root;
        KdNode next = null;
        while (p.left != null || p.right != null){
            if(query.getFeatureInDimension(dimension) < p.getFeatureInDimension(dimension)){
                next = p.left;
            }
            else if(query.getFeatureInDimension(dimension) > p.getFeatureInDimension(dimension)){
                next = p.right;
            }
            else {
                if(Double.compare(query.computeDistance(p.left), query.computeDistance(p.right)) <= 0){
                    next = p.left;
                }
                else {
                    next = p.right;
                }
            }
            if(next == null){
                break;
            }
            p = next;
            ++dimension;
            dimension %= root.data.length;
        }
        return p;
    }

    private void sortByDimension(List<KdNode> nodes, final int dimension){
        Collections.sort(nodes, new Comparator<KdNode>() {
            public int compare(KdNode o1, KdNode o2) {
                return Double.compare(o1.getFeatureInDimension(dimension), o2.getFeatureInDimension(dimension));
            }
        });
    }

    private void mainMaxHeap(List<KdNode> nodes, KdNode newNode, int k){
        if(nodes.size() < k){
            nodes.add(newNode);
            maxHeapFixUp(nodes);
        }
        else {
            if(Double.compare(newNode.distance, nodes.get(0).distance) < 0){
                nodes.set(0, newNode);
                maxHeapFixDown(nodes);
            }
        }
    }

    private void maxHeapFixUp(List<KdNode> nodes){
        int j = nodes.size()-1;
        int i = (j+1)/2 - 1;    //i:parent index of index j
        while (i>=0){
            if(Double.compare(nodes.get(i).distance, nodes.get(j).distance) >= 0){
                break;
            }
            KdNode t = nodes.get(i);
            nodes.set(i, nodes.get(j));
            nodes.set(j, t);
            j = i;
            i = (j+1)/2 -1;
        }
    }

    private void maxHeapFixDown(List<KdNode> nodes){
        int i = 0;
        int j = i*2+1;
        while (j < nodes.size()){
            if(j+1 < nodes.size() && Double.compare(nodes.get(j+1).distance, nodes.get(j).distance) > 0){
                ++j;
            }
            if(Double.compare(nodes.get(i).distance, nodes.get(j).distance) >= 0){
                break;
            }
            KdNode t = nodes.get(i);
            nodes.set(i, nodes.get(j));
            nodes.set(j, t);
            i = j;
            j = i*2 + 1;
        }
    }

    public void print(KdNode root){
        Queue<KdNode> q = new LinkedList<KdNode>();
        Queue<KdNode> nextLevel = new LinkedList<KdNode>();
        q.add(root);
        while (!q.isEmpty()) {
            while (!q.isEmpty()) {
                KdNode kdNode = q.remove();
                if(kdNode.left != null){
                    nextLevel.add(kdNode.left);
                }
                if(kdNode.right != null){
                    nextLevel.add(kdNode.right);
                }
                System.out.print(kdNode.toString() + "\t");
            }
            System.out.println();
            Queue<KdNode> t = q;
            q = nextLevel;
            nextLevel = t;
        }
    }

    @Test
    public void sortByFeatureDimesion(){
        List<KdNode> nodes = new ArrayList<KdNode>();
        nodes.add(new KdNode(new double[]{2, 3}));
        nodes.add(new KdNode(new double[]{5, 4}));
        nodes.add(new KdNode(new double[]{9, 6}));
        nodes.add(new KdNode(new double[]{4, 7}));
        nodes.add(new KdNode(new double[]{8, 1}));
        nodes.add(new KdNode(new double[]{7, 2}));
        sortByDimension(nodes, 1);
        for (KdNode node : nodes) {
            System.out.println(node.toString());
        }
    }

    @Test
    public void testBuildKdTree(){
        List<KdNode> nodes = new ArrayList<KdNode>();
        nodes.add(new KdNode(new double[]{2, 3}));
        nodes.add(new KdNode(new double[]{5, 4}));
        nodes.add(new KdNode(new double[]{9, 6}));
        nodes.add(new KdNode(new double[]{4, 7}));
        nodes.add(new KdNode(new double[]{8, 1}));
        nodes.add(new KdNode(new double[]{7, 2}));
        KdTree kdTree = new KdTree();
        KdNode root = kdTree.buildKdTree(nodes);
        kdTree.print(root);
    }

    @Test
    public void testKnnSearch(){
        List<KdNode> nodes = new ArrayList<KdNode>();
        nodes.add(new KdNode(new double[]{2, 3}));
        nodes.add(new KdNode(new double[]{5, 4}));
        nodes.add(new KdNode(new double[]{9, 6}));
        nodes.add(new KdNode(new double[]{4, 7}));
        nodes.add(new KdNode(new double[]{8, 1}));
        nodes.add(new KdNode(new double[]{7, 2}));
        KdTree kdTree = new KdTree();
        KdNode root = kdTree.buildKdTree(nodes);
        List<KdNode> knnNodes = kdTree.searchKnn(root, new KdNode(new double[]{2.1, 3.1}), 2);
        for (KdNode knnNode : knnNodes) {
            System.out.println(knnNode);
        }
    }

    static class KdNode implements Comparable<KdNode>{
        public double[] data;
        public KdNode left, right, parent;
        public double distance;
        public int dimension;

        public KdNode(double[] data){
            this.data = data;
        }
        public double getFeatureInDimension(int index){
            if(data == null || index >= data.length){
                return Integer.MAX_VALUE;
            }
            return data[index];
        }

        public int compareTo(KdNode o) {
            return Double.compare(distance, o.distance);
        }

        public double computeDistance(KdNode o){
            if(this.data == null || o.data == null || this.data.length != o.data.length){
                return Integer.MAX_VALUE;
            }
            double dis = 0;
            for (int i = 0; i < o.data.length; i++) {
                dis += Math.pow(getFeatureInDimension(i)-o.getFeatureInDimension(i), 2);
            }
            return Math.sqrt(dis);
        }

        public KdNode getBrother(){
            if(this == this.parent.left){
                return this.parent.right;
            }
            else {
                return this.parent.left;
            }
        }
        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < data.length; i++) {
                sb.append(data[i]);
                sb.append(" ");
            }
            return sb.toString();
        }
    }
}
