## Chatbot AI Assistant (Context Knownledge Base) - Top Knownledge Scores

#### Technology: OpenAi, OpenAi Embeddings, Cossine Similarity
#### Method: Completions (gpt-3.5-turbo model)

#### Description:
Chatbot developed with Python and Flask that features conversation with a virtual assistant. This uses a context based conversation and the answers are focused on a local datasets that are vectorised into embeddings and saved to `embeddings` directory. Every question asked is also vectorised and its embeddings are matched against the dataset embeddings to find three top most matches, then appended to a `system` prompt for `gpt-3.5-turbo` model. If the question isn't the first, last top match is also appended to `system` prompt to enable follow-up questions.

Note:
Embeddings are calculated when no embeddings matching file exist, if a file in `localdata` directory is modified that its embeddings are already calculated, it need to manually remove the matching `embeddings` directory file to remake the calculation.

### How to run (commands Windows terminal with Python 2.7):

#### Part One: Prepare Environment
- **Define necessary parameters (OpenAi API key, ...) on file 'qa_engine.py'**
- Initialize virtual environment and install dependencies, run:

	    virtualenv env
	    env\Scripts\activate
	    pip install flask python-dotenv
		python -m pip install -r requirements.txt

#### Part Two: Prepare local content
- Add documents to folder "localdata", do not use large files, split them if possible

#### Part Three: Run the app
- Initialize the app:

	    python app.py

- Enter "http://localhost:5000" on browser to interact with app

#### Changelog
- v0.1
	- initial build