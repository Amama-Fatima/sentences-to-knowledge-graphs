import modal
import json

class CharacterExtractionClient:
    def __init__(self):
        self.app = modal.App.lookup("character-extraction", create_if_missing=False)
        self.extract_function = modal.Function.lookup("character-extraction", "extract_characters_endpoint")
        self.test_function = modal.Function.lookup("character-extraction", "quick_test_endpoint")
    
    def extract_characters(self, story_text: str) -> dict:
        return self.extract_function.remote(story_text)
    
    def extract_from_file(self, file_path: str) -> dict:
        with open(file_path, 'r', encoding='utf-8') as f:
            story_text = f.read()
        return self.extract_characters(story_text)
    
    def quick_test(self) -> dict:
        return self.test_function.remote()

client = CharacterExtractionClient()

story_text = """
The first victim collapsed during the Tuesday morning board meeting.
Dr. Amanda Cross, Zenith Pharmaceuticals' lead researcher, had been mid-sentence explaining their breakthrough Alzheimer's drug when she suddenly clutched her throat and fell forward onto the conference table. Within minutes, she was dead. The autopsy would later reveal a rare botanical toxin—one that required sophisticated knowledge to synthesize.
By Friday, two more researchers were dead, and Detective Raj Patel found himself navigating the sterile corridors of one of Silicon Valley's most secretive biotech companies.
"""

print("\nExtracting characters...")
result = client.extract_characters(story_text)
print(json.dumps(result, indent=2))