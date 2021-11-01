import java.io.File;
import java.io.FileNotFoundException;
import java.lang.IndexOutOfBoundsException;
import java.util.Scanner;
import java.util.Set;
import java.util.List;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.Properties;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.simple.*;
import edu.stanford.nlp.trees.*;

/***
* @author Heng Zhang
* @since 2021-10-08
* This is a Java script extracts 20 features, which are specified in A2-3 description, from a given text corpus using coreNLP
 */

public class Sentence {
    /// 20 Features we want to extract from each sentence
    /// Details of each feature can be found in Word document A2-3Desciption
    String class_label, content; /// either ClauseAnaph or NomAnaph
    List<CoreLabel> tokens;
    int f1,f2,f3,f4,f5,f10,f15,f16, iterator;
    boolean f6,f8,f9,f11,f12,f13,f14,f17,f19,f20;
    String[] f7 = new String[8];
    ArrayList<String> f18 = new ArrayList<String>();
    Sentence other_it_ocurrances = null; ///Need to be checked for every Sentence Object in order to get full extraction of repeated "it"s

    Tree tree;                         /// Tree used for Chunking (Constituency Parsing) 
    Set<Constituent> treeConstituents; /// reference: https://nlp.stanford.edu/nlp/javadoc/javanlp-3.5.0/edu/stanford/nlp/trees/Tree.html



    /***
    * Constructor
    * @param List<CoreLabel> tokens   Content of this object (already tokenized)
    * @param int iterator            Indicates the number of "it" in this sentence that should be extracted
    * @version 3.0: Considered the case a sentence has multiple "it"
     */
    public Sentence(String class_label, List<CoreLabel> tokens, String content, Tree tree, int iterator){
        this.iterator = iterator;
        this.class_label = class_label;
        this.content = content;
        this.tokens = tokens;
        this.tree = tree;
        this.treeConstituents = tree.constituents(new LabeledScoredConstituentFactory()); ///reference: https://stanfordnlp.github.io/CoreNLP/parse.html
        int[] feature_1_to_3 = getF1_3();
        this.f1 = feature_1_to_3[0];
        this.f2 = feature_1_to_3[1];
        this.f3 = feature_1_to_3[2];
        int[] feature_4_to_5 = getF4_5();
        this.f4 = feature_4_to_5[0];
        this.f5 = feature_4_to_5[1];
        this.f6 = getF6();
        this.f7 = getF7();
        boolean[] feature_8_to_9 = getF8_9();
        this.f8 = feature_8_to_9[0];
        this.f9 = feature_8_to_9[1];
        this.f10 = getF10();
        this.f11 = getF11();
        boolean[] feature_12_to_13 = getF12_13();
        this.f12 = feature_12_to_13[0];
        this.f13 = feature_12_to_13[1];
        this.f14 = getF14();
        this.f15 = getF15();
        this.f16 = getF16();
        this.f17 = getF17();
    }

    public Sentence(String class_label, List<CoreLabel> tokens, String content, Tree tree){
        this(class_label, tokens, content, tree, 1);
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
                    this.other_it_ocurrances = new Sentence(this.class_label, this.tokens, this.content, this.tree, this.iterator+1);
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
    * @version 3.0
    */
    private int[] getF4_5(){
        int[] res = new int[2];
        int p_noun = 0;
        int a_noun = 0;
        ArrayList<Constituent> list_of_NPs = this.getPs("NP");
        for (Constituent constituent: list_of_NPs){
            if (constituent.start() < (this.f1 - 1)){   // Case: Noun phrase preceding-+
                p_noun ++;
            }else if (constituent.start() > (this.f1 - 1)){                                     // Case: Noun phrase after
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
    * @version 3.0
    */
    private boolean getF6(){
        ArrayList<Constituent> list_of_NPs = this.getPs("PP");
        for (Constituent constituent: list_of_NPs){
            if ((constituent.end() + 1) == (this.f1 - 1)){
                return true;
            }
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
    * Feature 8 and 9
    * Get whether occurrence of “it” followed by an -ing form of a verb
    * Get whether occurrence of “it” followed by a preposition
    * @return boolean[2]
    * @version 1.0
    */
    private boolean[] getF8_9(){
        String tag_of_word_follows_it = this.tokens.get(this.f1).tag();
        if (tag_of_word_follows_it.equals("VBG")){
            return new boolean[] {true, false};
        }else if (tag_of_word_follows_it.equals("IN")){
            return new boolean[] {false, true};
        }else{
            return new boolean[] {false, false};
        }
    }

     /***
    * Feature 10
    * Get the number of adjectives that follow the occurrence of “it” in the sentence.
    * @return int
    * @version 1.0
    */
    private int getF10(){
        int counter = 0;
        for (int x = this.f1; x < this.tokens.size(); x++){
            if (this.tokens.get(x).tag().contains("JJ")){
                counter ++;
            }
        }
        return counter;
    }

     /***
    * Feature 11
    * Check whether the pronoun “it” preceded by a verb
    * @return int
    * @version 1.0
    */
    private boolean getF11(){
        for (int x = 0; x < (this.f1-1); x++){
            if (this.tokens.get(x).tag().contains("VB")){
                return true;
            }
        }
        return false;
    }

    /***
    * Feature 12 and 13
    * F12	Is the pronoun “it” followed by a verb? (Yes/No)
    * F13	Is the pronoun “it” followed by an adjective? (Yes/No)
    * @return int
    * @version 1.0
    */
    private boolean[] getF12_13(){
        boolean[] res = new boolean[2];
        for (int x = this.f1; x < this.tokens.size(); x++){
            if (this.tokens.get(x).tag().contains("VB")){
                res[0] = true;
            }else if (this.tokens.get(x).tag().contains("JJ")){
                res[1] = true;
            }
        }
        return res;
    }

    /***
    * Feature 14
    * True if there is a noun phrase coming after the pronoun “it” 
    * and that noun phrase contains an adjective, otherwise false.
    * @return boolean
    * @version 1.0
     */
     private boolean getF14(){
        ArrayList<Constituent> list_of_NPs = this.getPs("NP");
        for (Constituent constituent: list_of_NPs){
           if (constituent.start() > (this.f1 - 1)){
                for (int x = constituent.start(); x <= constituent.end(); x++){
                    if(this.tokens.get(x).tag().contains("JJ")){
                        return true;
                    }
                }
            }
        }
        return false;
     }

     /***
    * Feature 15
    * The number of tokens coming before the following infinitive verb (if there is one), otherwise 0.
    * Based on my personal limited understanding of material, the target the following infinitive verb
    * means that the infinitive verb after "it"
    * @return int
    * @version 1.0
     */
    private int getF15(){
        /// the number of tokens coming before the infinitive verb = tokens before "it" + tokens between "it" and verb
        for (int counter = this.f1; counter < this.tokens.size() - 1; counter++){
            if (this.tokens.get(counter).word().toLowerCase().equals("to")){
                if (this.tokens.get(counter + 1).tag().equals("VB")){ /// infinitive verb found
                    return counter;
                }
            }
        }
        return 0;
    }

     /***
    * Feature 16
    * The number of tokens that appear between the pronoun “it” and the first following preposition 
    * (if there is a following preposition), otherwise 0.
    * @return int
    * @version 1.0
     */
    private int getF16(){
        for (int counter = this.f1; counter < this.tokens.size(); counter++){
            if (this.tokens.get(counter).tag().equals("IN")){
                return counter - this.f1;
            }
        }
        return 0;
    }

    /***
    * Feature 17
    * True if there a sequence “adjective + noun phrase” following the pronoun “it”, and false otherwise.
    *
    * I personally don't think this would happen in a grammatically correct sentence. ￣Д￣＝３
    * @return boolean
    * @version 1.0
     */
    private boolean getF17(){
        ArrayList<Constituent> list_of_NPs = this.getPs("NP");
        for (Constituent constituent: list_of_NPs){
           if (constituent.start() > (this.f1 - 1)){
                if (this.tokens.get(constituent.start() - 1).tag().contains("JJ")){
                    return true;
                }
            }
        }
        return false;
    }

    /***
    * Output Extracted Info. for each "it" word in sentence
    * @version 2.0: for testing propose
    */
    public void output(){
        System.out.println(this.content);
        System.out.println(this.class_label + "," +this.f1+ "," + this.f2+ "," + this.f3+ "," + this.f4+ "," + 
            this.f5+ "," + this.f6 + "," + Arrays.toString(this.f7) + "," + this.f8 + "," + this.f9+ "," + this.f10
            + "," + this.f11 + "," + this.f12 + "," + this.f13 + "," + this.f14+ "," + this.f15 + "," + this.f16 +
            "," + this.f17); //// Testing
        if (other_it_ocurrances != null){
            other_it_ocurrances.output();
        }
    }

    /***
    * Check whether a type of Phrase can be found, phrase type is specified as parameter; e.g. NP, VP
    * @param String phrase name
    * @return constituent[]
    * @version 2.0: Design for future reuse
    * @referencn https://nlp.stanford.edu/nlp/javadoc/javanlp-3.5.0/edu/stanford/nlp/trees/Constituent.html
    */
    public ArrayList<Constituent> getPs(String phrase_type){
        ArrayList<Constituent> res = new ArrayList<Constituent>();
        for (Constituent constituent : this.treeConstituents){
            if ( (constituent.label() != null) && (constituent.label().toString().equals(phrase_type))){
                if (constituent.size() != 0){ /// filter the word (phrases are considered composited of at least 2 words)

                    /// filter Phrases that are't atomic
                    boolean is_atomic = true;
                    for (Constituent elem : res){
                        if (elem.start() == constituent.start()){ // Case this constituent is not atomic (shares the same start with a shorter another)
                            is_atomic = false;
                        }
                    }
                    if(is_atomic){
                        res.add(constituent);
                    }
                }
            }
        }
        return res;
    }



    /*** 
    * Main Function
    * @param String[] args   args[0] is the file name
    * @return Nothing
    * @see FileNotFoundException
    * @version 3.0: Splitting Labels from Sentence, removing redundant sentences
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
            String[] sentence = scanner.nextLine().split("\t");
            if (!sentences.contains(sentence[1])){ /// Remove Redundant Sentences
                class_labels.add(sentence[0]);
                sentences.add(sentence[1]);
            }
        }

        /// NLP setup
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize, ssplit, pos, lemma, ner, parse");
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);

        /// Create Sentence objects, process them using CoreNLP one by one
        for (int x = 0; x < sentences.size(); x ++){
            CoreDocument to_be_tokenized = new CoreDocument(sentences.get(x));
            pipeline.annotate(to_be_tokenized); /// Sentences Tokenized 
            Sentence sentence = new Sentence(class_labels.get(x), to_be_tokenized.tokens(), sentences.get(x),
                to_be_tokenized.annotation().get(CoreAnnotations.SentencesAnnotation.class).get(0).get(TreeCoreAnnotations.TreeAnnotation.class));
            sentence.output();
        }
    }
}