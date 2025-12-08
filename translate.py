import json
import os
from pathlib import Path
from typing import Dict, List, Any
from deep_translator import GoogleTranslator, MyMemoryTranslator
import time
import re
from tqdm import tqdm

# Configuration
ROOT_DIR = "data"
LANGUAGES = [ "assamese"]
TRANSLATION_CACHE_FILE = "translation_cache.json"

# Load or initialize translation cache
def load_cache():
    if os.path.exists(TRANSLATION_CACHE_FILE):
        with open(TRANSLATION_CACHE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(TRANSLATION_CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def get_language_code(language: str) -> str:
    """Map language name to Google Translate language code"""
    lang_map = {
        "assamese": "as",
        "bengali": "bn",
        "spanish": "es"
    }
    return lang_map.get(language, "auto")

def clean_generated_answer(generated: str) -> str:
    """Clean generated answer by taking first token before comma"""
    if not generated:
        return ""
    
    if "," in generated:
        return generated.split(",")[0].strip()
    return generated.strip()

def clean_translated_text(text: str) -> str:
    """Clean translated text to extract main word/phrase"""
    if not text:
        return ""
    
    text = text.lower().strip()
    
    # Remove common filler patterns
    patterns = [
        r'^(a|an|the)\s+',
        r'\s+(is|are|was|were)\s+.*$',
        r'^it\s+(is|was)\s+',
        r'^this\s+(is|was)\s+',
        r'^that\s+(is|was)\s+',
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Take first word/phrase before punctuation or conjunctions
    text = re.split(r'[,;.]|\s+(is|are|and|or)\s+', text)[0].strip()
    
    return text

def translate_text(text: str, source_lang: str, cache: Dict) -> str:
    """Translate text with caching and cleaning"""
    if not text:
        return ""
    
    cache_key = f"{source_lang}:{text}"
    
    if cache_key in cache:
        return cache[cache_key]
    
    try:
        # Try Google Translator first
        translator = GoogleTranslator(source=source_lang, target='en')
        translated = translator.translate(text)
        
        # Clean the translation
        cleaned = clean_translated_text(translated)
        
        # If cleaning resulted in empty string, try MyMemory
        if not cleaned:
            try:
                translator_mm = MyMemoryTranslator(source=source_lang, target='en')
                translated = translator_mm.translate(text)
                cleaned = clean_translated_text(translated)
            except:
                cleaned = translated
        
        cache[cache_key] = cleaned
        time.sleep(0.1)  # Rate limiting
        return cleaned
        
    except Exception as e:
        print(f"Translation error for '{text}': {e}")
        return text

def process_file(input_path: str, output_path: str, language: str, cache: Dict):
    """Process a single JSONL file"""
    lang_code = get_language_code(language)
    processed_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        tqdm_iterator = tqdm(infile, desc=f"Translating {language}", unit="entry")
        total_lines = sum(1 for _ in open(input_path, 'r', encoding='utf-8'))
        tqdm_iterator.total = total_lines
        tqdm_iterator.refresh()

        for line in tqdm_iterator:
            data = json.loads(line)
            
            # Clean the generated answer
            cleaned_generated = clean_generated_answer(data.get("Generated", ""))
            data["CleanedAnswer"] = cleaned_generated
            
            # Translate cleaned generated answer
            if cleaned_generated:
                translated_generated = translate_text(cleaned_generated, lang_code, cache)
                data["TranslatedAnswer"] = translated_generated
            else:
                data["TranslatedAnswer"] = ""
            
            # Translate answer list
            translated_answers = []
            for answer in data.get("Answer", []):
                translated = translate_text(answer, lang_code, cache)
                translated_answers.append(translated)
                print(f"Original: {answer} | Translated: {translated}")
            
            data["TranslatedAnswerList"] = translated_answers
            
            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
            processed_count += 1
            
            if processed_count % 10 == 0:
                print(f"Processed {processed_count} entries for {language}")
                save_cache(cache)
    
    save_cache(cache)
    print(f"Completed {language}: {processed_count} entries")

def main():
    cache = load_cache()
    
    # Process each language
    for language in LANGUAGES:
        for file_type in ["generated"]:
            input_filename = f"predictions-{language}-{file_type}.jsonl"
            output_filename = f"predictions-{language}-{file_type}-translated.jsonl"
            
            input_path = os.path.join(ROOT_DIR, input_filename)
            output_path = os.path.join(ROOT_DIR, output_filename)
            
            if not os.path.exists(input_path):
                print(f"File not found: {input_path}")
                continue
            
            print(f"\nProcessing {input_filename}...")
            process_file(input_path, output_path, language, cache)
    
    print(f"\nTranslation complete. Cache saved to {TRANSLATION_CACHE_FILE}")
    print(f"Total cached translations: {len(cache)}")

if __name__ == "__main__":
    main()