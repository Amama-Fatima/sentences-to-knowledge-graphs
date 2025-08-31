import modal
from neo4j import GraphDatabase
from typing import List
import re
from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Neo4jConfig:
    """Configuration for Neo4j connection."""
    uri: str = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    username: str = os.getenv('NEO4J_USERNAME')
    password: str = os.getenv('NEO4J_PASSWORD')

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

class SimpleCharacterKnowledgeGraph:
    """Simple Neo4j knowledge graph with only character nodes."""
    
    def __init__(self, neo4j_config: Neo4jConfig):
        self.driver = GraphDatabase.driver(
            neo4j_config.uri, 
            auth=(neo4j_config.username, neo4j_config.password)
        )
    
    def close(self):
        """Close the Neo4j driver connection."""
        self.driver.close()
    
    def clear_database(self):
        """Clear all nodes and relationships in the database."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("Database cleared.")
    
    def generate_character_id(self, character_name: str) -> str:
        """
        Generate character ID following pattern: char_name_format
        Examples: char_dr._amanda_cross, char_detective_raj_patel
        """
        # Clean the name: remove special characters, convert to lowercase
        clean_name = re.sub(r'[^\w\s]', '', character_name.lower())
        # Replace spaces with underscores
        clean_name = re.sub(r'\s+', '_', clean_name.strip())
        # Handle titles like "Dr." -> "dr._"
        clean_name = clean_name.replace('dr_', 'dr._')
        
        return f"char_{clean_name}"
    
    def create_character_node(self, character: dict) -> str:
        """
        Create a character node with flexible schema.
        Only 'name' and 'role_in_story' are required fields.
        """
        # Required fields
        name = character.get('name', '').strip()
        role_in_story = character.get('role_in_story', 'character')
        
        if not name:
            raise ValueError("Character name is required")
        
        # Generate character ID
        char_id = self.generate_character_id(name)
        
        # Build dynamic properties - include all fields from character data
        properties = {
            'id': char_id,
            'name': name,
            'role_in_story': role_in_story
        }
        
        # Add all other properties dynamically
        for key, value in character.items():
            if key not in ['name', 'role_in_story'] and value:  # Skip empty values
                # Clean key name for Neo4j property
                clean_key = re.sub(r'[^\w]', '_', key)
                properties[clean_key] = str(value).strip() if value else ''
        
        prop_assignments = []
        for key in properties.keys():
            prop_assignments.append(f"{key}: ${key}")
        
        query = f"""
        MERGE (c:Character {{id: $id}})
        SET c += {{{', '.join(prop_assignments)}}}
        RETURN c.id, c.name
        """
        
        with self.driver.session() as session:
            result = session.run(query, properties)
            record = result.single()
            if record:
                print(f"Created/Updated character: {record['c.name']} (ID: {record['c.id']})")
                return char_id
            else:
                raise Exception(f"Failed to create character: {name}")
    
    def create_relationships_between_characters(self, characters: List[dict]):
        """Create relationships between characters based on their attributes."""
        
        # Create INVESTIGATES relationships (Detective -> Victim)
        detectives = [c for c in characters if 'detective' in c.get('role', '').lower() or 
                     c.get('role_in_story', '').lower() == 'detective']
        victims = [c for c in characters if c.get('status', '').lower() == 'dead' or 
                  c.get('role_in_story', '').lower() == 'victim']
        
        for detective in detectives:
            detective_id = self.generate_character_id(detective['name'])
            for victim in victims:
                victim_id = self.generate_character_id(victim['name'])
                
                query = """
                MATCH (d:Character {id: $detective_id})
                MATCH (v:Character {id: $victim_id})
                MERGE (d)-[:INVESTIGATES]->(v)
                """
                
                with self.driver.session() as session:
                    session.run(query, {
                        'detective_id': detective_id,
                        'victim_id': victim_id
                    })
                    print(f"Relationship: {detective['name']} INVESTIGATES {victim['name']}")
        
        # Create WORKS_AT relationships (characters at same location)
        location_groups = {}
        for char in characters:
            location = char.get('location', '').strip()
            if location and location.lower() not in ['', 'unknown', 'not mentioned']:
                if location not in location_groups:
                    location_groups[location] = []
                location_groups[location].append(char)
        
        # Create COLLEAGUE relationships
        for location, chars in location_groups.items():
            if len(chars) > 1:
                for i, char1 in enumerate(chars):
                    for char2 in chars[i+1:]:
                        char1_id = self.generate_character_id(char1['name'])
                        char2_id = self.generate_character_id(char2['name'])
                        
                        query = """
                        MATCH (c1:Character {id: $char1_id})
                        MATCH (c2:Character {id: $char2_id})
                        MERGE (c1)-[:COLLEAGUE]->(c2)
                        MERGE (c2)-[:COLLEAGUE]->(c1)
                        """
                        
                        with self.driver.session() as session:
                            session.run(query, {
                                'char1_id': char1_id,
                                'char2_id': char2_id
                            })
                            print(f"Relationship: {char1['name']} COLLEAGUE {char2['name']}")
    
    def process_extraction_result(self, extraction_result: dict, create_relationships: bool = True):
        """
        Process character extraction result and create knowledge graph.
        
        Args:
            extraction_result: Results from character extraction
            create_relationships: Whether to create relationships between characters
        """
        print("Processing characters into knowledge graph...")
        
        characters = extraction_result.get('characters', [])
        character_ids = []
        
        # Create character nodes
        for character in characters:
            try:
                char_id = self.create_character_node(character)
                character_ids.append(char_id)
            except Exception as e:
                print(f"Error creating character {character.get('name', 'Unknown')}: {e}")
                continue
        
        if create_relationships and len(characters) > 1:
            self.create_relationships_between_characters(characters)
        
        print(f"Successfully processed {len(character_ids)} characters")
        return character_ids
    
    def get_all_characters(self) -> List[dict]:
        """Get all characters from the database."""
        query = """
        MATCH (c:Character)
        RETURN c
        ORDER BY c.name
        """
        
        with self.driver.session() as session:
            result = session.run(query)
            characters = []
            for record in result:
                char_data = dict(record['c'])
                characters.append(char_data)
            return characters
    
    def get_character_relationships(self) -> List[dict]:
        """Get all relationships between characters."""
        query = """
        MATCH (c1:Character)-[r]->(c2:Character)
        RETURN c1.name as from_character,
               c1.id as from_id,
               type(r) as relationship_type,
               c2.name as to_character,
               c2.id as to_id
        """
        
        with self.driver.session() as session:
            result = session.run(query)
            relationships = []
            for record in result:
                relationships.append(dict(record))
            return relationships
    
    def query_characters(self, cypher_query: str) -> List[dict]:
        """Execute custom Cypher query."""
        with self.driver.session() as session:
            result = session.run(cypher_query)
            return [dict(record) for record in result]

class CharacterExtractionToKnowledgeGraph:
    """Combined client for character extraction and knowledge graph creation."""
    
    def __init__(self, neo4j_config: Neo4jConfig):
        self.extraction_client = CharacterExtractionClient()
        self.kg = SimpleCharacterKnowledgeGraph(neo4j_config)
    
    def process_story(self, story_text: str, clear_db: bool = False, create_relationships: bool = True) -> dict:
        """
        Extract characters from story and create knowledge graph.
        
        Args:
            story_text: The story text to process
            clear_db: Whether to clear database before processing
            create_relationships: Whether to create relationships between characters
            
        Returns:
            Dictionary with extraction results and character IDs
        """
        print("Step 1: Extracting characters...")
        extraction_result = self.extraction_client.extract_characters(story_text)
        print(f"Found {extraction_result.get('total_characters', 0)} characters")
        
        if clear_db:
            print("Step 2: Clearing database...")
            self.kg.clear_database()
        
        print("Step 3: Creating knowledge graph...")
        character_ids = self.kg.process_extraction_result(extraction_result, create_relationships)
        
        # Print summary
        print("\n=== KNOWLEDGE GRAPH SUMMARY ===")
        characters = self.kg.get_all_characters()
        for char in characters:
            print(f"• {char['name']} (ID: {char['id']}) - Role: {char.get('role_in_story', 'N/A')}")
        
        if create_relationships:
            relationships = self.kg.get_character_relationships()
            if relationships:
                print("\n=== RELATIONSHIPS ===")
                for rel in relationships:
                    print(f"• {rel['from_character']} --{rel['relationship_type']}--> {rel['to_character']}")
        
        return {
            'extraction_result': extraction_result,
            'character_ids': character_ids,
            'total_characters_in_kg': len(characters)
        }
    
    def close(self):
        """Close connections."""
        self.kg.close()

# Example usage
if __name__ == "__main__":
    config = Neo4jConfig()
    
    story_text = """
    The first victim collapsed during the Tuesday morning board meeting.
    Dr. Amanda Cross, Zenith Pharmaceuticals' lead researcher, had been mid-sentence explaining their breakthrough Alzheimer's drug when she suddenly clutched her throat and fell forward onto the conference table. Within minutes, she was dead. The autopsy would later reveal a rare botanical toxin—one that required sophisticated knowledge to synthesize.
    By Friday, two more researchers were dead, and Detective Raj Patel found himself navigating the sterile corridors of one of Silicon Valley's most secretive biotech companies.
    """
    
    processor = CharacterExtractionToKnowledgeGraph(config)
    
    try:
        # Process story
        result = processor.process_story(
            story_text=story_text,
            clear_db=True,
            create_relationships=True  # Create relationships between characters
        )
        
        print(f"\nSuccess! Created knowledge graph with {result['total_characters_in_kg']} characters")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure Neo4j is running and credentials are correct")
    
    finally:
        processor.close()