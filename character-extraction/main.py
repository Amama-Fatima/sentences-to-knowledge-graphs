import modal
import json
from typing import List, Dict
import torch

app = modal.App("character-extraction-test")

image = (
    modal.Image.debian_slim()
    .pip_install("torch>=2.0.0", index_url="https://download.pytorch.org/whl/cu121")  # for utilizing GPU instead of CPU
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

def chunk_story_spacy(story_text: str, max_sentences: int = 8, target_words: int = 150) -> List[str]:
    """Smaller chunks for faster processing."""
    import spacy

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

def extract_characters_from_chunk(chunk_text: str, model, tokenizer, chunk_number: int) -> str:
    """Fast character extraction with optimized settings."""
    prompt = f"""<|im_start|>system
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
    
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        truncation=True, 
        max_length=512
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Aggressive optimization for speed
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=250,
            temperature=0.1, 
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            repetition_penalty=1.05,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("RESPONSE IS ", response)
    return response

def parse_character_json(raw_response: str) -> List[Dict]:
    """Simple JSON parsing."""
    try:
        start_idx = raw_response.find('[')
        end_idx = raw_response.rfind(']') + 1
        
        if start_idx != -1 and end_idx != 0:
            json_str = raw_response[start_idx:end_idx]
            # json_str = json_str.replace("'", '"')
            characters = json.loads(json_str)
            
            if isinstance(characters, dict):
                characters = [characters]
            
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

volume = modal.Volume.from_name("openhermes-cache", create_if_missing=True)

@app.function(
    image=image,
    gpu="A100",
    memory=32768,  # Reduced memory
    timeout=600,   # 10 minutes
    min_containers=1,
)
def extract_characters_fast(story_text: str) -> dict:
    """Fast character extraction with optimized model loading."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import os
    import time
    
    start_time = time.time()
    print("Starting fast character extraction...")
    
    os.environ['TRANSFORMERS_CACHE'] = '/cache'
    os.environ['HF_HOME'] = '/cache'
    
    print("Loading OpenHermes models with optimizations...")
    model_name = "teknium/OpenHermes-2.5-Mistral-7B"
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Optimized model loading
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    model.eval()
    
    load_time = time.time() - start_time
    print(f"Models loaded in {load_time:.2f} seconds")
    
    # Process in small chunks for speed
    chunks = chunk_story_spacy(story_text, max_sentences=8, target_words=150)
    print(f"Story split into {len(chunks)} chunks")
    
    all_characters = []
    
    for i, chunk in enumerate(chunks, 1):
        chunk_start = time.time()
        print(f"Processing chunk {i}/{len(chunks)}...")
        
        try:
            raw_response = extract_characters_from_chunk(chunk, model, tokenizer, i)
            chunk_characters = parse_character_json(raw_response)
            all_characters.extend(chunk_characters)
            
            chunk_time = time.time() - chunk_start
            print(f"Chunk {i} completed in {chunk_time:.2f}s - found {len(chunk_characters)} characters")
            
        except Exception as e:
            print(f"Error processing chunk {i}: {str(e)}")
            continue
    
    unique_characters = []
    seen_names = set()
    
    for char in all_characters:
        name_lower = char.get("name", "").lower().strip()
        if name_lower and name_lower not in seen_names and len(name_lower) > 1:
            seen_names.add(name_lower)
            unique_characters.append(char)
    
    total_time = time.time() - start_time
    print(f"Extraction complete! Found {len(unique_characters)} characters in {total_time:.2f} seconds")
    
    return {
        'characters': unique_characters,
        'chunk_count': len(chunks),
        'total_characters': len(unique_characters),
        'processing_time_seconds': total_time,
        'model_load_time_seconds': load_time,
        'model_used': model_name
    }

@app.function(
    image=image,
    gpu="A100", 
    timeout=300,
    min_containers=1,
)
def quick_test() -> dict:
    """Quick test with minimal generation."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import os
    import time
    
    start = time.time()
    os.environ['TRANSFORMERS_CACHE'] = '/cache'
    
    tokenizer = AutoTokenizer.from_pretrained("teknium/OpenHermes-2.5-Mistral-7B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        "teknium/OpenHermes-2.5-Mistral-7B",
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    model.eval()
    
    # Minimal test
    prompt = "Extract names: Dr. Smith works here."
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    gen_start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=0.5,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    gen_time = time.time() - gen_start
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    total = time.time() - start
    
    return {
        "response": response,
        "generation_time": gen_time,
        "total_time": total
    }

@app.local_entrypoint()
def main(story_file: str = None):
    """Main function with your story."""
    if story_file:
        with open(story_file, 'r', encoding='utf-8') as f:
            story_text = f.read()
    else:
        story_text = """The first victim collapsed during the Tuesday morning board meeting.
Dr. Amanda Cross, Zenith Pharmaceuticals' lead researcher, had been mid-sentence explaining their breakthrough Alzheimer's drug when she suddenly clutched her throat and fell forward onto the conference table. Within minutes, she was dead. The autopsy would later reveal a rare botanical toxin—one that required sophisticated knowledge to synthesize.
By Friday, two more researchers were dead, and Detective Raj Patel found himself navigating the sterile corridors of one of Silicon Valley's most secretive biotech companies."""
    
    print("Starting fast character extraction...")
    result = extract_characters_fast.remote(story_text)
    print(json.dumps(result, indent=2))

@app.local_entrypoint()
def test():
    """Quick test function."""
    print("Running quick test...")
    result = quick_test.remote()
    print(json.dumps(result, indent=2))