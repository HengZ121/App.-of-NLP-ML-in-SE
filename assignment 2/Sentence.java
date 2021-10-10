import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;
import java.util.ArrayList;
import java.util.Properties;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.simple.*;

/***
* @author Heng Zhang
* @since 2021-10-08
 */

public class Sentence {
    /// 20 Features we want to extract from each sentence
    /// Details of each feature can be found in Word document A2-3Desciption
    CoreDocument sentence;
    int f1,f2,f3,f4,f5,f10,f15,f16;
    boolean f6,f8,f9,f11,f12,f13,f14,f17,f19,f20;
    ArrayList<String> f7 = new ArrayList<String>(8);
    ArrayList<String> f18 = new ArrayList<String>();
    


    /***
    * Constructor
    * @param CoreDocument sentence   Content of this object (already tokenized)
     */
    public Sentence(CoreDocument sentence){
        this.sentence = sentence;
        int[] feature_1_to_3 = getF1_3();
        this.f1 = feature_1_to_3[0];
        this.f2 = feature_1_to_3[1];
        this.f3 = feature_1_to_3[2];
    }

    /***
    * Feature 1 to 3
    * Get the position of "it" in the sentence considering the number of tokens
    * Get the length of the sentence in terms of tokens
    * Get the number of punctuations
    * @return int[5]   res[0] = -1 if no "it" in sentence; res[1] = -1 if sentence is invalid
    * @version 2.0
    */
    private int[] getF1_3(){
        int[] res = {-1,-1,0};
        int counter = 0;
        int punctuations = 0;
        
        for (CoreLabel tok : this.sentence.tokens()) {
            if (tok.word().toLowerCase().equals("it")){
                res[0] = counter; /// Position of "it" F1 done
            }else if (Pattern.matches("\\p{Punct}", tok.word()) || tok.word().equals("...")){/// Use Pattern to check punctuations
                punctuations ++;
            }
            counter ++;
        }
        res[1] = counter - 1; /// F2 done
        res[2] = punctuations; // F3 done
        return res;
    }

    /***
    * Feature 4 and 5
    * Get the number of preceding noun phrases
    * Get the number of noun phrases after "it"
    * @return int[5]    -1 if no "it" in sentence
    * @version 1.0
    */
    

    /*** 
    * Main Function
    * @param String[] args   args[0] is the file name
    * @return Nothing
    * @see FileNotFoundException
    */ 
    public static void main(String[] args) throws FileNotFoundException{

        /// Let user indicates file contains target sentences via args.
        File dataset = new File(args[0]);
        /// Read sentences and store them in an ArrayList
        Scanner scanner = new Scanner(dataset);
        ArrayList<String> sentences = new ArrayList<String>();
        scanner.nextLine();
        while (scanner.hasNextLine()){
            String sentence = scanner.nextLine();
            sentences.add(sentence);
        }

        /// NLP setup
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize, pos");
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);

        /// Create Sentence objects, process them using CoreNLP one by one
        for (String elem : sentences){
            CoreDocument to_be_tokenized = new CoreDocument(elem);
            pipeline.annotate(to_be_tokenized); /// Sentences Tokenized 
            Sentence sentence = new Sentence(to_be_tokenized);
            System.out.println(elem + " " + sentence.f1+ " " + sentence.f2+ " " + sentence.f3); //// Testing
        }
    }
}