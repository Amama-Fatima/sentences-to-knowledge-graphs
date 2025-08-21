import os
import warnings

import torch
original_load = torch.load

# The patched_load() function is a fix for PyTorch 2.6+ weights_only issue.

def patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_load(*args, **kwargs)

torch.load = patched_load

warnings.filterwarnings("ignore", message=".*weights_only.*")

import stanza
from neo4j import GraphDatabase
from dotenv import load_dotenv
import logging
from nltk.stem import WordNetLemmatizer
import nltk

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SentenceToKG:
    def __init__(self):
        """Initialize the NLP pipeline and Neo4j connection."""
        print("🚀 Starting initialization...")
        logger.info("Starting SentenceToKG initialization")
        
        load_dotenv() 
        print("✅ Environment variables loaded")
        
        print("📦 Downloading NLTK data...")
        try:
            nltk.download('wordnet', quiet=True)
            print("✅ NLTK wordnet downloaded successfully")
        except Exception as e:
            logger.warning(f"NLTK wordnet download failed: {e}, but continuing...")
            print("⚠️ NLTK wordnet download failed, but continuing...")
        
        # Initialize Stanza NLP pipeline
        print("🧠 Initializing Stanza NLP model (this may take a while)...")
        logger.info("Initializing Stanza NLP model...")
        try:
            print("   - Creating pipeline...")
            self.nlp = stanza.Pipeline(
                'en', 
                processors='tokenize,pos,lemma,depparse',
                use_gpu=False,
                logging_level='ERROR'
            )
            logger.info("Stanza pipeline initialized successfully.")
            print("✅ Stanza pipeline initialized successfully!")
        except Exception as e:
            logger.error(f"Failed to initialize Stanza pipeline: {e}")
            print(f"❌ Failed to initialize Stanza pipeline: {e}")
            # alternative approaches if the first one fails
            try:
                print("   - Trying alternative approach...")
                logger.info("Trying alternative approach...")
                # Force download fresh models
                print("   - Downloading fresh Stanza models...")
                stanza.download('en')
                print("   - Creating pipeline with fresh models...")
                self.nlp = stanza.Pipeline(
                    'en', 
                    processors='tokenize,pos,lemma,depparse',
                    use_gpu=False,
                    logging_level='ERROR'
                )
                logger.info("Stanza pipeline initialized successfully with fresh download.")
                print("✅ Stanza pipeline initialized successfully with fresh download!")
            except Exception as e2:
                logger.error(f"All initialization attempts failed: {e2}")
                print(f"❌ All initialization attempts failed: {e2}")
                raise
        
        # Initialize Neo4j driver
        print("🔌 Setting up Neo4j connection...")
        self.uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.user = os.getenv('NEO4J_USERNAME', 'neo4j')
        self.password = os.getenv('NEO4J_PASSWORD', '')
        
        print(f"   - Neo4j URI: {self.uri}")
        print(f"   - Neo4j User: {self.user}")
        
        if not self.password:
            logger.error("Neo4J_PASSWORD not found in .env file. Please check your configuration.")
            print("❌ Neo4J_PASSWORD not found in .env file. Please check your configuration.")
            print("⚠️ Continuing without Neo4j connection...")
            self.driver = None
        else:
            try:
                print("   - Attempting to connect to Neo4j...")
                self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
                print("   - Testing Neo4j connection...")
                with self.driver.session() as session:
                    session.run("RETURN 1 AS test")
                logger.info("Neo4j connection established successfully.")
                print("✅ Neo4j connection established successfully!")
            except Exception as e:
                logger.error(f"Failed to connect to Neo4j: {e}")
                print(f"❌ Failed to connect to Neo4j: {e}")
                print("⚠️ Continuing without Neo4j connection for demo purposes...")
                self.driver = None
        
        # Initialize lemmatizer for verb normalization
        print("📝 Initializing lemmatizer...")
        try:
            self.lemmatizer = WordNetLemmatizer()
            print("✅ Lemmatizer initialized successfully!")
        except Exception as e:
            print(f"❌ Failed to initialize lemmatizer: {e}")
            self.lemmatizer = None
        
        print("🎉 Initialization complete!")
    
    def get_verb_lemma(self, word_text, stanza_lemma, pos_tag):
        """
        Get the best lemma for a word, with fallback strategies.
        """
        # Some common lemmatization errors I faced. Will probably remove this in the future
        lemma_override_dict = {
            "hugge": "hug", "hugged": "hug",
            "ate": "eat", "ran": "run",
            "was": "be", "were": "be",
            "did": "do", "had": "have",
            "is": "be", "are": "be", "am": "be"
        }
        
        if stanza_lemma in lemma_override_dict:
            return lemma_override_dict[stanza_lemma]
        
        # Use NLTK's WordNet lemmatizer for verbs
        if pos_tag.startswith('V') and self.lemmatizer:  # If it's a verb and lemmatizer exists
            try:
                nltk_lemma = self.lemmatizer.lemmatize(word_text, pos='v')
                if nltk_lemma != stanza_lemma and len(nltk_lemma) < 6:  # Simple heuristic
                    return nltk_lemma
            except:
                pass  # Fall back to stanza lemma if NLTK fails
        
        return stanza_lemma
    
    def extract_triples(self, sentence):
        """
        Extract subject-verb-object triples from a sentence using dependency parsing.
        Returns a list of triples: (subject, relation, object)
        """
        logger.info(f"Processing sentence: '{sentence}'")
        
        try:
            doc = self.nlp(sentence)
        except Exception as e:
            logger.error(f"Failed to process sentence with Stanza: {e}")
            return []
        
        triples = []
        
        for sent in doc.sentences:
            # Find the root verb (main action)
            root_verb = None
            for word in sent.words:
                if word.head == 0 and word.upos.startswith('V'):  # It's a root verb
                    root_verb = word
                    break
            
            if not root_verb:
                logger.warning(f"No root verb found in sentence: {sentence}")
                continue
            
            # Get subjects and objects related to this root verb
            subjects = []
            objects = []
            
            for word in sent.words:
                if word.head == root_verb.id:
                    if word.deprel in ['nsubj', 'nsubj:pass']:  # nominal subject
                        subjects.append(word)
                    elif word.deprel == 'obj':  # direct object
                        objects.append(word)
            
            verb_lemma = self.get_verb_lemma(
                root_verb.text, 
                root_verb.lemma, 
                root_verb.upos
            ).upper()
            
            # triples for each subject-object pair
            for subj in subjects:
                for obj in objects:
                    triples.append((
                        subj.text,      # Subject
                        verb_lemma,     # Relation
                        obj.text        # Object
                    ))
        
        logger.info(f"Extracted {len(triples)} triple(s) from sentence")
        return triples
    
    def create_kg_from_triples(self, triples):
        """
        Create nodes and relationships in Neo4j from extracted triples.
        """
        if not triples:
            logger.warning("No triples to add to knowledge graph")
            return
        
        if not self.driver:
            logger.warning("No Neo4j connection available. Skipping database operations.")
            return
        
        with self.driver.session() as session:
            for subject, relation, obj in triples:
                # MERGE to avoid duplicates
                query = """
                MERGE (s:Entity {name: $subject})
                MERGE (o:Entity {name: $object})
                MERGE (s)-[r:%s]->(o)
                SET r.last_updated = timestamp()
                RETURN s, r, o
                """ % relation
                
                try:
                    result = session.run(query, subject=subject, object=obj)
                    count = result.consume().counters.relationships_created
                    if count > 0:
                        logger.info(f"Created relationship: ({subject})-[:{relation}]->({obj})")
                    else:
                        logger.info(f"Relationship already exists: ({subject})-[:{relation}]->({obj})")
                except Exception as e:
                    logger.error(f"Failed to create relationship: {e}")
    
    def process_sentence(self, sentence):
        """
        Full pipeline: process a sentence and add it to the knowledge graph.
        """
        try:
            triples = self.extract_triples(sentence)
            self.create_kg_from_triples(triples)
            return triples
        except Exception as e:
            logger.error(f"Error processing sentence: {e}")
            return []
    
    def close(self):
        """Close the Neo4j driver connection."""
        if hasattr(self, 'driver') and self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed.")

def main():
    """Main function to run the application."""
    print("🔧 Starting Knowledge Graph Builder...")
    kg_builder = None
    try:
        kg_builder = SentenceToKG()
        
        print("=" * 60)
        print("Knowledge Graph Builder Started!")
        print("Type 'quit' to exit the application.")
        print("=" * 60)
        
        while True:
            try:
                sentence = input("\nEnter a sentence to add to the knowledge graph: ").strip()
            except EOFError:
                print("\nGoodbye!")
                break
            
            if sentence.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not sentence:
                continue
            
            triples = kg_builder.process_sentence(sentence)
            
            if triples:
                print("\n✅ Extracted Triples:")
                for i, (s, r, o) in enumerate(triples, 1):
                    print(f"  {i}. ({s}) -[{r}]-> ({o})")
            else:
                print("❌ No triples could be extracted from this sentence.")
                
    except KeyboardInterrupt:
        print("\nApplication interrupted by user.")
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if kg_builder:
            kg_builder.close()

if __name__ == "__main__":
    main()