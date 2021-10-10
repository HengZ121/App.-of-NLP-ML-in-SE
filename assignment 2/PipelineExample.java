package edu.stanford.nlp.examples;

import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.pipeline.*;

import java.util.*;

public class PipelineExample {

  public static String text = "Marie was born in Paris.";

  public static void main(String[] args) {
    // set up pipeline properties
    Properties props = new Properties();
    // set the list of annotators to run
    props.setProperty("annotators", "tokenize,ssplit,pos");
    // build pipeline
    StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
    // create a document object
    CoreDocument document = pipeline.processToCoreDocument(text);
    // display tokens
    for (CoreLabel tok : document.tokens()) {
      System.out.println(String.format("%s\t%s", tok.word(), tok.tag()));
    }
  }
}
