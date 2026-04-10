#This sets up the python virtual environment that we'll use to run tests.
#But Ollama still has to be installed separately and run in the background before we can do anything.

export TMPDIR="/tmp"

VENV_DIR="llama_venv"

if [ ! -d $VENV_DIR ]; then
    echo "Making environment..."
    python3 -m venv $VENV_DIR
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip
    pip install --no-cache-dir llama-index llama-index-llms-ollama llama-index-embeddings-huggingface
    pip install --no-cache-dir pypdf reportlab python-docx
    echo "Virtual environment is all set up!"
else
    echo "Virtual environment is already set up."
fi

echo "Use \"source $VENV_DIR/bin/activate\" to enter it."

