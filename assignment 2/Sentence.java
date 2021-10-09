import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;
import java.util.ArrayList;
import java.util.Properties;
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
        this.f1 = getF1();
    }

    /***  
    * Get the position of "it" in the sentence considering the number of tokens
    * @return int   The position of "it" in the sentence; return -1 if no "it" is found
    * @version 1.0: comma or other punctuation marks is also assigned a token
    */
    private int getF1(){
        int pos = 0; ///sentence[0] indicates Class of sentence and has value either NomAnaph or ClauseAnaph
        for (CoreLabel tok : this.sentence.tokens()) {
            if (tok.word().toLowerCase().equals("it")){
                return pos;
            }else{
                pos ++;
            }
        }
        return -1;
    }


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
        props.setProperty("annotators", "tokenize");
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);

        /// Create Sentence objects, process them using CoreNLP one by one
        for (String elem : sentences){
            CoreDocument to_be_tokenized = new CoreDocument(elem);
            pipeline.annotate(to_be_tokenized); /// Sentences Tokenized 
            Sentence sentence = new Sentence(to_be_tokenized);
            System.out.println(elem + " " + sentence.f1); //// Testing
        }
    }
}