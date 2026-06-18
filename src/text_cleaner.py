import re

def clean_subtitle_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    
    # 1. Remove HTML tags like <i>, </i>, <br>
    # <[^>]+> matches anything starting with < and ending with >
    text = re.sub(r'<[^>]+>', '', text)
    
    # 2. Remove standard English parentheses and everything inside them: (Name)
    text = re.sub(r'\([^)]*\)', '', text)
    
    # 3. Remove Japanese full-width parentheses and everything inside them: （名前）
    text = re.sub(r'（[^）]*）', '', text)
    
    # 4. Clean up any accidental double spaces left behind by the deletions
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
