# document-classification
Automatic Document Classification based on Image Analysis

This is a model for identifying the document type in an automated way (e.g. email, scientific publication, memo, etc).
The model has been tested on the RVL-CDIP dataset, which is available at: http://www.cs.cmu.edu/~aharley/rvl-cdip/

Installation

To install the required libraries (tested on Ubuntu 17.11) run:

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Classify documents

Training the model from scratch

Prepare a dataset:

Place your document images in a folder and set the folder as the "dataset_path" in AutoDocClass.py
Create a train, test and validation set CSV file containing "Relative Filename Path, Document Class id" pairs and "Filename, Class" as the header row

Train the model:

Set the model parameters in AutoDocClass.py and run the script


In progress:
    - Main file callable from the terminal with tunable model parameters
    - Callable model "export to disk" function 
