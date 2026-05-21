# Assignment: Building Your Own Transformer

## Objective

- The main.py script should run for at least one epoch without errors and output test performance results.
- The model's checkpoint should be saved, and when a sentence is inputted, the model should generate an output.

## Rules

- Upload the project to the "transformer_from_scratch" folder in the CS224N GitHub repository.
- Set up a virtual environment and include the dependency package list in a requirements.txt file.
- Do not use pre-trained models from Hugging Face! Preferably, use PyTorch-based implementations.
- Hyperparameters should be based on the "Attention Is All You Need" paper.
- The project structure should be as follows:
  - model.py: Contain functions required for the model to operate and integrate various components.
  - main.py: Calls the model to train and evaluate it.
  - requirements.txt: Lists the required Python libraries for execution (see Rule 2).
  - readme.md: Provides a general guide on running the model.

## Resources

- Dataset: Use wmt/wmt19 from Hugging Face Datasets.
  - Split the validation set: use half as dev_set and the other half as test_set.
- Tokenizer: Any pre-existing tokenizer from existing models can be used.
- Training and Evaluation:
  - Loss function: Cross-entropy loss
  - Optimizer: Adam
  - Evaluation metrics: Perplexity, BLEU

---

## Environments

- The virtual environment used for this assignment can be reconstructed using `requirements.txt` and following command:
  ```bash
  conda create --name transformer_from_scratch --file requirements.txt
  ```
