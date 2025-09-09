Lab 1 notes

- [Links](#links)
- [Highlights](#highlights)
- [Admin](#admin)
  - [Working with colab \& github](#working-with-colab--github)
  - [Magic commands](#magic-commands)

# Links
- spacy: https://spacy.io/usage/spacy-101
- nltk: https://www.nltk.org/api/nltk.tokenize.html
- hugging face transformer and tokenizer: https://huggingface.co/docs/transformers/main_classes/tokenizer
- pytorch: https://pytorch.org/
- on rules and different languages: http://ceur-ws.org/Vol-2226/paper9.pdf
- why do we need language-specific tokenisation: https://stackoverflow.com/questions/17314506/why-do-i-need-a-tokenizer-for-each-language
  - brief answer: not all languages separate tokens/linguistically meaningful units by whitespaces

# Highlights
- tokenization
  - spacy package
  - ntlk package - seems to perform better

- transformers 
  - hugging face package
    - for training, using, and deploying transformer-based models 

  - pytorch 
    - machine learning framework 
    - used for building & training neural networks (esp. in deep learning)


# Admin
Working with colab & github
-----
- you can open notebooks hosted in GitHub on Colab, this will open a new editable version of the notebook and any changes won't override the GitHub version. If you want to save the changes to GitHub select File->Make a copy to GitHub.

- can work in vs code, but if it takes a long time to work, upload to Google Colab (with my T4 GPU) - it will load much faster when we are training transformers 


Magic commands
-----
- special commands in IPython or Jupyter Notebooks.
- an extra feature that helps with interactive work, debugging, timing, and system tasks

Magic used in this course : 

- automatically re-load imported modules every time before executing the Python code typed
- if you use Colab, just comment out the following two lines
%load_ext autoreload
%autoreload 2

- automatically include matplotlib plots in the frontend of the notebook and save them with the notebook
%matplotlib inline


