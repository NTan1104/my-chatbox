#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to clean and fix Vietnamese language errors in CV markdown files.
Fixes include:
- Missing spaces after punctuation
- Incorrect spacing around Vietnamese words
- Standardizing common Vietnamese typos
- Fixing formatting issues
"""

import os
import re
from pathlib import Path

# Common Vietnamese typos and fixes
VIETNAMESE_FIXES = {
    r'\(([A-Za-z])': r' (\1',  # Add space before opening parenthesis
    r'(\d)\s*-\s*(\d)': r'\1 - \2',  # Normalize dashes with spaces
    'tuỳ biến': 'tùy biến',  # Common typo
    'tuỳ chỉnh': 'tùy chỉnh',  # Common typo
    'Vịtrí': 'Vị trí',  # Missing space
    '⁕': '•',  # Normalize bullet points
    'Nhiệm vụ': 'Nhiệm vụ',  # This one is OK, just ensuring consistency
    'Công nghệ thông tin': 'Công Nghệ Thông Tin',  # Consistency
}

# More sophisticated regex-based fixes
REGEX_FIXES = [
    # Remove various Unicode icons and symbols
    (r'⭐\s*', ''),  # Remove star icons
    (r'●\s*', ''),   # Remove filled circle icons
    (r'○\s*', ''),   # Remove hollow circle icons
    (r'◆\s*', ''),   # Remove diamond icons
    (r'■\s*', ''),   # Remove square icons
    (r'✓\s*', ''),   # Remove checkmark icons
    (r'✔\s*', ''),   # Remove check icons
    (r'✕\s*', ''),   # Remove cross icons
    (r'✗\s*', ''),   # Remove X icons
    (r'★\s*', ''),   # Remove star outline
    (r'☆\s*', ''),   # Remove hollow star
    (r'※\s*', ''),   # Remove reference mark
    (r'❏\s*', ''),   # Remove ballot box
    (r'❑\s*', ''),   # Remove ballot box
    (r'⁕\s*', ''),   # Remove special bullet point
    (r'▪\s*', ''),   # Remove small square
    (r'▫\s*', ''),   # Remove small hollow square
    (r'►\s*', ''),   # Remove right-pointing triangle
    (r'▸\s*', ''),   # Remove small right-pointing triangle
    (r'→\s*', ''),   # Remove right arrow
    (r'←\s*', ''),   # Remove left arrow
    (r'↔\s*', ''),   # Remove left-right arrow
    (r'⚬\s*', ''),   # Remove medium small white circle
    (r'◉\s*', ''),   # Remove fisheye
    
    # Add space between Vietnamese word and opening parenthesis if missing
    (r'([a-záàảãạăằẳẵặâầẩẫậéèẻẽẹêềếểễệíìỉĩịóòỏõọôồốổỗộơờớởỡợúùủũụưừứửữựỳỵỷỹýđ])\(', r'\1 ('),
    
    # Fix "Vịtrí" -> "Vị trí" (OCR error common in Vietnamese)
    (r'\bVịtrí\b', 'Vị trí'),
    
    # Fix common OCR issues with punctuation marks
    (r'\s+(\.|,|:|;|\?|!)', r'\1'),  # Remove space before punctuation
    
    # Normalize multiple spaces
    (r'  +', ' '),  # Multiple spaces to single space
]

# Lines that should be preserved with their original formatting
PRESERVE_PATTERNS = [
    r'©.*',  # Copyright lines
    r'^.*http.*$',  # URLs
    r'^.*@.*$',  # Email addresses
    r'^\d+\.$',  # Line numbers
]

def should_preserve_line(line):
    """Check if a line should be preserved as-is."""
    for pattern in PRESERVE_PATTERNS:
        if re.search(pattern, line):
            return True
    return False

def fix_vietnamese_text(text):
    """Apply Vietnamese-specific fixes to text."""
    
    # Apply simple string replacements first
    for old, new in VIETNAMESE_FIXES.items():
        if old in text:
            text = text.replace(old, new)
    
    # Apply regex-based fixes
    for pattern, replacement in REGEX_FIXES:
        text = re.sub(pattern, replacement, text, flags=re.UNICODE | re.MULTILINE)
    
    # Fix spacing issues after common abbreviations and words
    text = re.sub(r'([A-Z]\.)\s*-\s*', r'\1 - ', text)  # "Ts. - " format
    
    # Fix Vietnamese number spacing (e.g., "0962984651" stays as is, but "12/12/2001" is OK)
    # Don't touch phone numbers, keep them as they are
    
    # Fix spacing around special Vietnamese diacritics
    text = re.sub(r'(\d)\s*-\s*(\d)', r'\1 - \2', text)  # Dates
    
    # Normalize spaces around parentheses
    text = re.sub(r'\(\s+', '(', text)  # Remove space after opening parenthesis
    text = re.sub(r'\s+\)', ')', text)  # Remove space before closing parenthesis
    
    # Fix common spacing issues with numbers
    text = re.sub(r'(\d)\s+\)', r'\1)', text)  # Remove space before closing paren after digit
    
    return text

def clean_markdown_file(filepath):
    """Clean a markdown file and fix Vietnamese language errors."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Preserve certain lines (URLs, emails, copyright)
            if should_preserve_line(line):
                cleaned_lines.append(line)
            else:
                # Apply cleaning
                cleaned_line = fix_vietnamese_text(line)
                # Additional cleanup: remove extra spaces
                cleaned_line = cleaned_line.strip()
                cleaned_lines.append(cleaned_line)
        
        cleaned_content = '\n'.join(cleaned_lines)
        
        # Only write if content changed
        if cleaned_content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)
            return True
        return False
    
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def clean_markdown_file_to_output(input_filepath, output_filepath):
    """Clean a markdown file and save to output location."""
    try:
        with open(input_filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Preserve certain lines (URLs, emails, copyright)
            if should_preserve_line(line):
                cleaned_lines.append(line)
            else:
                # Apply cleaning
                cleaned_line = fix_vietnamese_text(line)
                # Additional cleanup: remove extra spaces
                cleaned_line = cleaned_line.strip()
                cleaned_lines.append(cleaned_line)
        
        cleaned_content = '\n'.join(cleaned_lines)
        
        # Write to output directory
        with open(output_filepath, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
        return True
    
    except Exception as e:
        print(f"Error processing {input_filepath}: {e}")
        return False

def process_directory(input_directory, output_directory):
    """Process all .md files in a directory and save as .txt to output."""
    md_files = list(Path(input_directory).glob('*.md'))
    
    if not md_files:
        print(f"No .md files found in {input_directory}")
        return
    
    # Create output directory if it doesn't exist
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    
    print(f"Found {len(md_files)} markdown files to process")
    print(f"Output directory: {output_directory}\n")
    
    processed = 0
    
    for filepath in sorted(md_files):
        # Convert .md to .txt in output
        output_filename = filepath.stem + '.txt'  # Gets filename without extension, adds .txt
        output_path = Path(output_directory) / output_filename
        clean_markdown_file_to_output(filepath, output_path)
        print(f"✓ Cleaned: {filepath.name} → {output_filename}")
        processed += 1
    
    print(f"\n{'='*50}")
    print(f"Summary:")
    print(f"  Total files processed: {processed}")
    print(f"  Output location: {output_directory}")
    print(f"  Output format: .txt files")

if __name__ == '__main__':
    # Path to the directory containing CV markdown files
    input_directory = r'd:\Projects\XAI\data\extracted\docling\md'
    output_directory = r'd:\Projects\XAI\data\cleaned'
    
    # Verify input directory exists
    if not os.path.exists(input_directory):
        print(f"Error: Directory not found: {input_directory}")
        exit(1)
    
    print(f"Starting Vietnamese text cleanup")
    print(f"  Input:  {input_directory}")
    print(f"  Output: {output_directory}\n")
    
    process_directory(input_directory, output_directory)
    
    print(f"\nCleaning complete!")
