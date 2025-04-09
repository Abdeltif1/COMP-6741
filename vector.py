import json
import csv
import re

# Load the JSON data
with open('concordia_cs_program_raw.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Extract main content
main_content = data.get('content', '')

# Function to clean text by removing special characters and extra whitespace
def clean_text(text):
    # Replace Unicode characters
    text = text.replace('\u00e2\u0080\u0099', "'")
    text = text.replace('\u00e2\u0080\u0093', "-")
    text = text.replace('\u00e2\u0080\u0099ll', "'ll")
    text = text.replace('\u00c3\u00a7', "ç")
    text = text.replace('\u00c3\u00a9', "é")
    text = text.replace('\u00c3\u00a8', "è")
    
    # Remove markdown links
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    
    # Remove markdown formatting
    text = re.sub(r'#{1,6}\s+', '', text)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Extract program details
program_info = {
    "Program Name": "Computer Science (BCompSc)",
    "University": "Concordia University",
    "Faculty": "Gina Cody School of Engineering and Computer Science",
    "Department": "Computer Science and Software Engineering",
    "Duration": "3 to 4 years",
    "Campus": "Sir George Williams (SGW)",
    "Start Terms": "Fall, Winter",
    "Experiential Learning": "Co-op"
}

# Extract admission requirements
admission_requirements = {}
if 'subpages' in data and 'https://www.concordia.ca/academics/undergraduate/computer-science.html#requirements' in data['subpages']:
    content = data['subpages']['https://www.concordia.ca/academics/undergraduate/computer-science.html#requirements']['content']
    
    # Extract CEGEP requirements
    cegep_match = re.search(r'\*\*Quebec CEGEP:\*\*\s+([^*]+)', content)
    if cegep_match:
        admission_requirements["CEGEP"] = clean_text(cegep_match.group(1))
    
    # Extract High School requirements
    hs_match = re.search(r'\*\*High School\*\*\s*:\s+([^*]+)', content)
    if hs_match:
        admission_requirements["High School"] = clean_text(hs_match.group(1))
    
    # Extract IB requirements
    ib_match = re.search(r'\*\*International Baccalaureate \(IB\) diploma:\*\*\s+([^*]+)', content)
    if ib_match:
        admission_requirements["IB"] = clean_text(ib_match.group(1))

# Extract program structure
program_structure = {}
if 'subpages' in data and 'https://www.concordia.ca/academics/undergraduate/computer-science.html#structure' in data['subpages']:
    content = data['subpages']['https://www.concordia.ca/academics/undergraduate/computer-science.html#structure']['content']
    
    # Extract core courses
    core_match = re.search(r'\*\*Core courses\*\*\s+(.*?)\*\s\*\s\*', content, re.DOTALL)
    if core_match:
        program_structure["Core Courses"] = clean_text(core_match.group(1))
    
    # Extract electives
    electives_match = re.search(r'\*\*Electives\*\*\s+(.*?)\*\s\*\s\*', content, re.DOTALL)
    if electives_match:
        program_structure["Electives"] = clean_text(electives_match.group(1))

# Extract career options
career_options = ""
if 'subpages' in data and 'https://www.concordia.ca/academics/undergraduate/computer-science.html#after' in data['subpages']:
    content = data['subpages']['https://www.concordia.ca/academics/undergraduate/computer-science.html#after']['content']
    
    career_match = re.search(r'## After your degree\s+(.*?)##', content, re.DOTALL)
    if career_match:
        career_options = clean_text(career_match.group(1))

# Create CSV file
with open('concordia_cs_program.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    
    # Write header
    writer.writerow(['Category', 'Detail'])
    
    # Write program info
    for key, value in program_info.items():
        writer.writerow([key, value])
    
    # Write admission requirements
    writer.writerow(['', ''])
    writer.writerow(['ADMISSION REQUIREMENTS', ''])
    for key, value in admission_requirements.items():
        writer.writerow([key, value])
    
    # Write program structure
    writer.writerow(['', ''])
    writer.writerow(['PROGRAM STRUCTURE', ''])
    for key, value in program_structure.items():
        writer.writerow([key, value])
    
    # Write career options
    writer.writerow(['', ''])
    writer.writerow(['CAREER OPTIONS', career_options])

print("CSV file 'concordia_cs_program.csv' has been created successfully.")