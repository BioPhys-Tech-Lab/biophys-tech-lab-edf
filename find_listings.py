import re

def find_listings(filename):
    with open(filename, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    listings = []
    current_listing = None
    
    for i, line in enumerate(lines):
        line_num = i + 1
        if '\\begin{lstlisting}' in line:
            caption_match = re.search(r'caption=\{(.*?)\}|caption=(.*?)\]', line)
            caption = "Unknown"
            if caption_match:
                caption = caption_match.group(1) or caption_match.group(2)
            
            current_listing = {'start': line_num, 'caption': caption}
        
        elif '\\end{lstlisting}' in line and current_listing:
            current_listing['end'] = line_num
            listings.append(current_listing)
            current_listing = None

    for l in listings:
        print(f"Listing: '{l['caption']}' | Lines: {l['start']} - {l['end']}")

find_listings('c:/Users/ASUS/Downloads/biophys-tech-lab-edf/collaboration.tex')
