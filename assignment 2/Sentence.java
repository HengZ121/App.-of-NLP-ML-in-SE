import java.io.File;
import java.io.FileNotFoundException;
import java.lang.IndexOutOfBoundsException;
import java.util.Scanner;
import java.util.List;
import java.util.Arrays;
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
    String class_label; /// either ClauseAnaph or NomAnaph
    List<CoreLabel> tokens;
    int f1,f2,f3,f4,f5,f10,f15,f16, iterator;
    boolean f6,f8,f9,f11,f12,f13,f14,f17,f19,f20;
    String[] f7 = new String[8];
    ArrayList<String> f18 = new ArrayList<String>();
    Sentence other_it_ocurrances = null; ///Need to be checked for every Sentence Object in order to get full extraction of repeated "it"s
    


    /***
    * Constructor
    * @param List<CoreLabel> tokens   Content of this object (already tokenized)
    * @param int iterator            Indicates the number of "it" in this sentence that should be extracted
    * @version 2.0: Considered the case a sentence has multiple "it"
     */
    public Sentence(String class_label, List<CoreLabel> tokens, int iterator){
        this.iterator = iterator;
        this.class_label = class_label;
        this.tokens = tokens;
        int[] feature_1_to_3 = getF1_3();
        this.f1 = feature_1_to_3[0];
        this.f2 = feature_1_to_3[1];
        this.f3 = feature_1_to_3[2];
        int[] feature_4_to_5 = getF4_5();
        this.f4 = feature_4_to_5[0];
        this.f5 = feature_4_to_5[1];
        this.f6 = getF6();
        this.f7 = getF7();
    }

    public Sentence(String class_label, List<CoreLabel> tokens){
        this(class_label, tokens, 1);
    }

    /***
    * Feature 1 to 3
    * Get the position of "it" in the sentence considering the number of tokens
    * Get the length of the sentence in terms of tokens
    * Get the number of punctuations
    * @return int[3]   res[0] = 0 if no "it" in sentence; res[1] = 0 if sentence is invalid or empty
    * @version 3.0
    */
    private int[] getF1_3(){
        int[] res = {0,0,0};
        int counter = 1;
        int punctuations = 0;
        int number_of_it_found = 0;
        
        for (CoreLabel tok : this.tokens) {
            if (tok.word().toLowerCase().equals("it")){ /// Found it
                number_of_it_found ++;
                if (number_of_it_found == this.iterator){ /// Found current iteration
                    res[0] = counter; /// Position of "it" F1 done
                }else if(number_of_it_found == (this.iterator + 1)){
                    other_it_ocurrances = new Sentence(this.class_label, this.tokens, this.iterator+1);
                }
                
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
    * @return int[2]
    * @version 1.0
    */
    private int[] getF4_5(){
        int[] res = new int[2];
        int p_noun = 0;
        int a_noun = 0;
        for (int x = 0; x < this.f1; x++){
            if (this.tokens.get(x).tag().contains("NN")){
                p_noun ++;
            }
        }
        for (int x = this.f1; x < this.tokens.size()-1; x++){
            if (this.tokens.get(x).tag().contains("NN")){
                a_noun ++;
            }
        }
        res[0] = p_noun;
        res[1] = a_noun;
        return res;
    }

    /***
    * Feature 6
    * Test whether the pronoun “it” immediately follow a prepositional phrase
    * @return boolean
    * @version 1.0
    */
    private boolean getF6(){
        if (this.f1 == this.tokens.size()){ /// Case: It is the last word
            return false;
        }
        if (this.tokens.get(this.f1).tag().equals("IN")){
            return true;
        }
        return false;
    }

    /***
    * Feature 7
    * Tags of the four tokens immediately preceding and the four tokens immediately succeeding a given instance of “it”. 
    * ABS (absent) to the missing POS tags
    * @return String[]
    * @version 1.0
    */
    private String[] getF7(){
        String[] res = new String[8];
        for (int x = this.f1 - 5; x < (this.f1 - 1); x++){    //// TAGS PRECEDING
            if (x >= 0){                           /// Case: Index x in boundary
                res[x - this.f1 +5] = this.tokens.get(x).tag();
            }else{                                 /// Case: Index x out of boundary, Missing POS tag found
                res[x - this.f1 +5] = "ABS";
            }
        }
        for (int x = this.f1; x < (this.f1 + 4); x ++){ //// TAGS SUCCEEDING
            try{                                   /// Case: Index x in boundary
                res[x - this.f1 + 4] = this.tokens.get(x).tag();
            }catch(IndexOutOfBoundsException e){   /// Case: Index x out of boundary, Missing POS tag found
                res[x - this.f1 + 4] = "ABS";
            }
        }
        return res;
    }

    /***
    * Output Extracted Info. for each "it" word in sentence
    * @version1.0: for testing propose
    */
    public void output(){
        for (CoreLabel elem: this.tokens){
            System.out.print(elem.word()+" ");
        }
        System.out.println();
        System.out.println(this.class_label + "," +this.f1+ "," + this.f2+ "," + this.f3+ "," + this.f4+ "," + 
            this.f5+ "," + this.f6 + "," + Arrays.toString(this.f7)); //// Testing
        if (other_it_ocurrances != null){
            other_it_ocurrances.output();
        }
    }
    

    /*** 
    * Main Function
    * @param String[] args   args[0] is the file name
    * @return Nothing
    * @see FileNotFoundException
    * @version 2.0: Splitting Labels from Sentence
    */ 
    public static void main(String[] args) throws FileNotFoundException{

        /// Let user indicates file contains target sentences via args.
        File dataset = new File(args[0]);
        /// Read sentences and store them in an ArrayList
        Scanner scanner = new Scanner(dataset);
        ArrayList<String> sentences = new ArrayList<String>();
        ArrayList<String> class_labels = new ArrayList<String>();
        scanner.nextLine();
        while (scanner.hasNextLine()){
            String sentence[] = scanner.nextLine().split("\t");
            class_labels.add(sentence[0]);
            sentences.add(sentence[1]);
        }

        /// NLP setup
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize, ssplit, pos");
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);

        /// Create Sentence objects, process them using CoreNLP one by one
        for (int x = 0; x < sentences.size(); x ++){
            CoreDocument to_be_tokenized = new CoreDocument(sentences.get(x));
            pipeline.annotate(to_be_tokenized); /// Sentences Tokenized 
            Sentence sentence = new Sentence(class_labels.get(x), to_be_tokenized.tokens());
            sentence.output();
        }
    }
}