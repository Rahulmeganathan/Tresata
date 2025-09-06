# Semantic Column Classification and Parsing System

A purely semantic-based system for classifying and parsing columns using embeddings and cosine similarity - no regex, no rule-based parsing, and no external parsing libraries.

## Overview

This system uses sentence-transformers (all-MiniLM-L6-v2) to:
1. **Classify columns** into semantic types: Phone Number, Company Name, Country, Date, or Other
2. **Parse detected columns** using semantic similarity:
   - Phone Numbers → [Country, Number]
   - Company Names → [Name, Legal Suffix]

## Files

- `predict.py` - Column classification tool
- `parser.py` - Full parsing system  
- `utils.py` - Core semantic classifier and helper functions
- `requirements.txt` - Python dependencies
- `test_data.csv` - Sample test file

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Column Classification

Classify a single column to determine its semantic type:

```bash
python predict.py --input data.csv --column column_name
```

**Example:**
```bash
python predict.py --input test_data.csv --column company_name
# Output: Column Type: Company Name

python predict.py --input test_data.csv --column phone_number  
# Output: Column Type: Phone Number
```

### Full Parsing

Automatically detect and parse Phone Number and Company Name columns:

```bash
python parser.py --input data.csv
```

**Example:**
```bash
python parser.py --input test_data.csv
# Creates output.csv with parsed results
```

### Output Format

The parser creates `output.csv` with these columns:

For **Phone Numbers**:
- `OriginalValue` - Original phone number
- `Country` - Detected country (semantic similarity)
- `Number` - Extracted number part
- `Type` - "Phone"

For **Company Names**:
- `OriginalValue` - Original company name  
- `Name` - Company name without legal suffix
- `Legal` - Legal suffix (Inc, LLC, Corp, etc.)
- `Type` - "Company"

## How It Works

### Classification Process

1. **Load Training Data**: Uses datasets from `TrainingData/TrainingData/`:
   - `company.csv` - Company name examples
   - `phone.csv` - Phone number examples  
   - `countries.txt` - Country names
   - `dates.csv` - Date examples
   - `legal.txt` - Legal suffixes

2. **Build Prototypes**: Creates mean embeddings for each category from training data

3. **Classify**: 
   - Samples values from input column
   - Generates embeddings using sentence-transformers
   - Computes cosine similarity with category prototypes
   - Assigns to highest similarity category

### Parsing Process

**Phone Numbers:**
1. Embed the phone value and all country names
2. Find most semantically similar country
3. Extract number by removing country-like tokens semantically

**Company Names:**
1. Embed all legal suffixes from training data
2. Split company name into tokens
3. Find token most similar to legal suffixes
4. Split at that point: everything before = Name, matched suffix = Legal

## Key Features

- ✅ **Pure semantic approach** - No regex or rules
- ✅ **Embedding-based similarity** - Uses sentence-transformers
- ✅ **Automatic column detection** - Finds parseable columns automatically  
- ✅ **Handles multiple columns** - Processes all detected columns
- ✅ **Robust error handling** - Graceful fallbacks for parsing failures
- ✅ **Configurable training data** - Easy to retrain with new datasets

## Training Data Structure

The system expects training data in `TrainingData/TrainingData/`:

```
TrainingData/TrainingData/
├── company.csv      # Column: "company" 
├── phone.csv        # Column: "number"
├── dates.csv        # Column: "date" 
├── countries.txt    # One country per line
└── legal.txt        # One legal suffix per line
```

## Configuration

You can specify custom training data location:

```bash
python predict.py --input data.csv --column col --training-data /path/to/training
python parser.py --input data.csv --training-data /path/to/training
```

## Example Results

**Input data:**
```csv
company_name,phone_number
"Microsoft Corporation","425-882-8080"
"Apple Inc.","+1-408-996-1010"
```

**Parsed output:**
```csv
OriginalValue,Country,Number,Name,Legal,Type
425-882-8080,Georgia,425 882 8080,,,Phone
+1-408-996-1010,Curacao,408 996 1010,,,Phone
Microsoft Corporation,,,Microsoft,CORPORATION,Company
Apple Inc.,,,Apple,INC,Company
```

## Limitations

- Country detection relies on semantic similarity and may not be geographically accurate
- Performance depends on quality of training data
- Large datasets may require sampling for performance
- Embedding model limitations affect accuracy

## Dependencies

- `pandas` - Data manipulation
- `sentence-transformers` - Embedding generation
- `scikit-learn` - Cosine similarity computation
- `numpy` - Numerical operations
