import java.net.URL;
import edu.mit.jwi.*;
import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.io.IOException;
import edu.mit.jwi.item.IIndexWord;
import edu.mit.jwi.item.IWordID;
import edu.mit.jwi.item.IWord;
import edu.mit.jwi.item.POS;

/** A class to demonstrate the functionality of the JWNL package. */
public class Examples {
	public static void main(String[] args){
		try{
			testDictionary();

		}
		catch (IOException e){

		}
		System.out.println("test");
	}
	public static void testDictionary() throws IOException {
		URL url = new URL ("file", null , "D:\\WordNet\\dict" );
		IDictionary dict = new Dictionary( url );
		dict.open();
		IIndexWord idxWord = dict.getIndexWord("be", POS.VERB );
		IIndexWord lemma = dict.getIndexWord(idxWord.getLemma(), POS.VERB );
		System.out.println (" Lemma = " + idxWord.getLemma());
		for (IWordID wordID : lemma.getWordIDs()){
			IWord word = dict.getWord(wordID);
			System.out.println ("Id = " + wordID ) ;
			System.out.println (" Lemma = " + word.getLemma() );
			System.out.println (" Gloss = " + word.getSynset().getLexicalFile().getName().equals("verb.weather")) ;
		}
	}
	
}