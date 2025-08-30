import modal
import json
import torch
import time
import os
from typing import List, Dict, Any

# ========================================
# CONFIGURATION
# ========================================

class Config:
    # Model Configuration
    MODEL_NAME = "teknium/OpenHermes-2.5-Mistral-7B"
    MODEL_DTYPE = "float16"
    CACHE_DIR = "/cache"
    
    # Generation Parameters
    MAX_NEW_TOKENS = 250
    TEMPERATURE = 0.1
    TOP_P = 0.9
    REPETITION_PENALTY = 1.05
    DO_SAMPLE = True
    USE_CACHE = True
    
    # Input Parameters
    MAX_INPUT_LENGTH = 512
    
    # Chunking Parameters
    MAX_SENTENCES_PER_CHUNK = 8
    TARGET_WORDS_PER_CHUNK = 150
    
    # Modal Configuration
    GPU_TYPE = "A100"
    MEMORY_MB = 32768
    TIMEOUT_SECONDS = 600
    MIN_CONTAINERS = 1
    
    # Application Settings
    APP_NAME = "character-extraction"
    VOLUME_NAME = "openhermes-cache"

# ========================================
# IMAGE BUILDER
# ========================================

def build_image() -> modal.Image:
    """Build the Modal image with all dependencies."""
    return (
        modal.Image.debian_slim()
        .pip_install("torch>=2.0.0", index_url="https://download.pytorch.org/whl/cu121")
        .pip_install("transformers==4.35.0")
        .pip_install("accelerate>=0.20.0") 
        .pip_install("sentencepiece>=0.1.99")
        .pip_install("protobuf>=3.20.0")
        .pip_install("spacy>=3.7.0")
        .pip_install("fastapi[standard]")
        .run_commands("python -m spacy download en_core_web_sm")
        # Pre-download the model during image build
        .run_commands(
            "mkdir -p /cache && "
            "python -c \""
            "from transformers import AutoTokenizer, AutoModelForCausalLM; "
            "import os; "
            "os.environ['TRANSFORMERS_CACHE'] = '/cache'; "
            "os.environ['HF_HOME'] = '/cache'; "
            "print('Downloading tokenizer...'); "
            "tokenizer = AutoTokenizer.from_pretrained('teknium/OpenHermes-2.5-Mistral-7B'); "
            "print('Downloading model...'); "
            "model = AutoModelForCausalLM.from_pretrained('teknium/OpenHermes-2.5-Mistral-7B', torch_dtype='auto'); "
            "print('Model pre-download complete!')"
            "\""
        )
    )

# ========================================
# TEXT PROCESSING
# ========================================

def chunk_story_spacy(
    story_text: str, 
    max_sentences: int = None, 
    target_words: int = None
) -> List[str]:
    """Split story into smaller chunks using spaCy sentence segmentation."""
    import spacy
    
    max_sentences = max_sentences or Config.MAX_SENTENCES_PER_CHUNK
    target_words = target_words or Config.TARGET_WORDS_PER_CHUNK
    
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(story_text)
    
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for sentence in sentences:
        word_count = len(sentence.split())
        
        if (len(current_chunk) >= max_sentences or 
            (current_word_count + word_count > target_words and current_chunk)):
            
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_word_count = word_count
        else:
            current_chunk.append(sentence)
            current_word_count += word_count
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    chunks = [chunk.replace("\n", " ").strip() for chunk in chunks]
    return chunks

# ========================================
# PROMPT TEMPLATES
# ========================================

class PromptTemplates:
    
    @staticmethod
    def character_extraction_prompt(chunk_text: str) -> str:
        """Generate character extraction prompt."""
        return f"""<|im_start|>system
You are an expert at extracting character information from stories. Extract ALL characters mentioned in the text, including their names, roles, and any relevant details. DO NOT give duplicate keys.

Create a detailed JSON object with these fields:
- "name": the character's full name
- "role": their job, position, or function in the story
- "status": alive/dead/arrested/missing/unknown
- "physical_description": any physical details mentioned in the story
- "location": where they work or are found
- "expertise": their skills or knowledge areas
- "role_in_story": victim/detective/witness/side-character

Important: Extract EVERY named character, regardless of how briefly they appear.
<|im_end|>
<|im_start|>user
{chunk_text}
<|im_end|>
<|im_start|>assistant
["""

    @staticmethod
    def simple_test_prompt(text: str) -> str:
        """Generate simple test prompt."""
        return f"Extract names: {text}"

# ========================================
# MODEL HANDLER
# ========================================

class ModelHandler:
    def __init__(self, config=Config):
        self.config = config
        self.tokenizer = None
        self.model = None
    
    def setup_cache(self):
        """Set up model cache directories."""
        os.environ['TRANSFORMERS_CACHE'] = self.config.CACHE_DIR
        os.environ['HF_HOME'] = self.config.CACHE_DIR
    
    def load_model(self):
        """Load tokenizer and model."""
        from transformers import AutoTokenizer, AutoModelForCausalLM

        print(f"Loading {self.config.MODEL_NAME}...")

        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.MODEL_NAME)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.MODEL_NAME,
            torch_dtype=getattr(torch, self.config.MODEL_DTYPE),
            device_map="auto",
        )
        
        self.model.eval()
        print("Model loaded successfully!")
    
    def generate_response(self, prompt: str, **generation_kwargs) -> str:
        """Generate response from model."""
        # Prepare generation parameters
        gen_params = {
            'max_new_tokens': self.config.MAX_NEW_TOKENS,
            'temperature': self.config.TEMPERATURE,
            'do_sample': self.config.DO_SAMPLE,
            'top_p': self.config.TOP_P,
            'repetition_penalty': self.config.REPETITION_PENALTY,
            'use_cache': self.config.USE_CACHE,
            'pad_token_id': self.tokenizer.eos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
        }
        
        # Override with any provided parameters
        gen_params.update(generation_kwargs)
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=self.config.MAX_INPUT_LENGTH
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_params)
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

# ========================================
# CHARACTER PARSER
# ========================================

class CharacterParser:
    
    @staticmethod
    def parse_character_json(raw_response: str) -> List[Dict]:
        """Parse character data from model response."""
        try:
            start_idx = raw_response.find('[')
            end_idx = raw_response.rfind(']') + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = raw_response[start_idx:end_idx]
                characters = json.loads(json_str)
                
                if isinstance(characters, dict):
                    characters = [characters]
                
                # Add missing fields with defaults
                for char in characters:
                    if isinstance(char, dict):
                        char.setdefault("status", "unknown")
                        char.setdefault("physical_description", "")
                        char.setdefault("location", "")
                        char.setdefault("expertise", "")
                        char.setdefault("role_in_story", "character")
                        char.setdefault("other_details", "")
                
                return characters
            
            return []
        
        except Exception as e:
            print(f"JSON parsing failed: {str(e)}")
            return []
    
    @staticmethod
    def deduplicate_characters(characters: List[Dict]) -> List[Dict]:
        """Remove duplicate characters based on name similarity."""
        unique_characters = []
        seen_names = set()
        
        for char in characters:
            name_lower = char.get("name", "").lower().strip()
            if name_lower and name_lower not in seen_names and len(name_lower) > 1:
                seen_names.add(name_lower)
                unique_characters.append(char)
        
        return unique_characters

# ========================================
# CHARACTER EXTRACTOR
# ========================================

class CharacterExtractor:
    
    def __init__(self, config=Config):
        self.config = config
        self.model_handler = ModelHandler(config)
        self.parser = CharacterParser()
    
    def extract_from_chunk(self, chunk_text: str, chunk_number: int) -> List[Dict]:
        """Extract characters from a single text chunk."""
        print(f"Processing chunk {chunk_number}...")
        
        prompt = PromptTemplates.character_extraction_prompt(chunk_text)
        raw_response = self.model_handler.generate_response(prompt)
        
        # Optional: print response for debugging
        # print(f"Raw response: {raw_response[:200]}...")
        
        characters = self.parser.parse_character_json(raw_response)
        return characters
    
    def extract_characters(self, story_text: str) -> Dict:
        """Extract characters from full story text."""
        start_time = time.time()
        print("Starting character extraction...")
        
        # Setup and load model
        self.model_handler.setup_cache()
        load_start = time.time()
        self.model_handler.load_model()
        load_time = time.time() - load_start
        
        # Process story in chunks
        chunks = chunk_story_spacy(story_text)
        print(f"Story split into {len(chunks)} chunks")
        
        all_characters = []
        
        for i, chunk in enumerate(chunks, 1):
            chunk_start = time.time()
            
            try:
                chunk_characters = self.extract_from_chunk(chunk, i)
                all_characters.extend(chunk_characters)
                
                chunk_time = time.time() - chunk_start
                print(f"Chunk {i} completed in {chunk_time:.2f}s - found {len(chunk_characters)} characters")
                
            except Exception as e:
                print(f"Error processing chunk {i}: {str(e)}")
                continue
        
        # Remove duplicates
        unique_characters = self.parser.deduplicate_characters(all_characters)
        
        total_time = time.time() - start_time
        print(f"Extraction complete! Found {len(unique_characters)} characters in {total_time:.2f} seconds")
        
        return {
            'characters': unique_characters,
            'chunk_count': len(chunks),
            'total_characters': len(unique_characters),
            'processing_time_seconds': total_time,
            'model_load_time_seconds': load_time,
            'model_used': self.config.MODEL_NAME
        }

# ========================================
# MODAL APPLICATION
# ========================================

# Initialize Modal app and image
app = modal.App(Config.APP_NAME)
image = build_image()
volume = modal.Volume.from_name(Config.VOLUME_NAME, create_if_missing=True)

@app.function(
    image=image,
    gpu=Config.GPU_TYPE,
    memory=Config.MEMORY_MB,
    timeout=Config.TIMEOUT_SECONDS,
    min_containers=Config.MIN_CONTAINERS,
)
def extract_characters_endpoint(story_text: str) -> dict:
    """Main character extraction endpoint."""
    extractor = CharacterExtractor()
    return extractor.extract_characters(story_text)

@app.function(
    image=image,
    gpu=Config.GPU_TYPE,
    timeout=300,
    min_containers=1,
)
def quick_test_endpoint() -> dict:
    """Quick test endpoint for debugging."""
    start = time.time()
    
    model_handler = ModelHandler()
    model_handler.setup_cache()
    model_handler.load_model()
    
    prompt = PromptTemplates.simple_test_prompt("Dr. Smith works here.")
    
    gen_start = time.time()
    response = model_handler.generate_response(
        prompt, 
        max_new_tokens=20,
        temperature=0.5
    )
    gen_time = time.time() - gen_start
    
    total = time.time() - start
    
    return {
        "response": response,
        "generation_time": gen_time,
        "total_time": total
    }

# Web API endpoint
@app.function(image=image, gpu=Config.GPU_TYPE, timeout=Config.TIMEOUT_SECONDS)
@modal.fastapi_endpoint(method="POST")
def web_extract_characters(request_data: dict):
    """Web API endpoint for character extraction."""
    story_text = request_data.get('story_text', '')
    if not story_text:
        return {"error": "No story_text provided"}
    
    extractor = CharacterExtractor()
    result = extractor.extract_characters(story_text)
    return result

@app.local_entrypoint()
def main(story_file: str = None):
    """Main CLI entry point."""
    if story_file:
        with open(story_file, 'r', encoding='utf-8') as f:
            story_text = f.read()
    else:
        story_text = """The first victim collapsed during the Tuesday morning board meeting.
Dr. Amanda Cross, Zenith Pharmaceuticals' lead researcher, had been mid-sentence explaining their breakthrough Alzheimer's drug when she suddenly clutched her throat and fell forward onto the conference table. Within minutes, she was dead. The autopsy would later reveal a rare botanical toxin—one that required sophisticated knowledge to synthesize.
By Friday, two more researchers were dead, and Detective Raj Patel found himself navigating the sterile corridors of one of Silicon Valley's most secretive biotech companies."""
    
    print("Starting character extraction...")
    result = extract_characters_endpoint.remote(story_text)
    print(json.dumps(result, indent=2))

@app.local_entrypoint()
def test():
    """Test CLI entry point."""
    print("Running quick test...")
    result = quick_test_endpoint.remote()
    print(json.dumps(result, indent=2))