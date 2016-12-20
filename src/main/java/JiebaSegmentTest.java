import com.huaban.analysis.jieba.JiebaSegmenter;
import com.huaban.analysis.jieba.SegToken;

/**
 * Created by yangwenjie on 16/12/19.
 */
public class JiebaSegmentTest {
    public static void main(String[] args) {
        String sentence = "这是一个伸手不见五指的黑夜。我叫孙悟空，我爱北京，我爱Python和C++。";
        JiebaSegmenter jiebaSegmenter = new JiebaSegmenter();
        /*for (SegToken segToken : jiebaSegmenter.process(sentence, JiebaSegmenter.SegMode.INDEX)) {
            System.out.println(segToken.word);
        }*/
        System.out.println(jiebaSegmenter.process(sentence, JiebaSegmenter.SegMode.INDEX));
    }
}
