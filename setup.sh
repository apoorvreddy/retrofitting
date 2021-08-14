conda env create -f environment.yml
python -c "import nltk; nltk.download('wordnet')"
pip install python-Levenshtein
python -m spacy download en_core_web_md