# Text Classification using Mamba with IMDB Dataset

## Overview
This project aims to perform sentiment analysis on the IMDB movie review dataset using the Mamba Model. The goal is to classify movie reviews as either positive or negative based on their textual content.

## Dataset
The IMDB dataset consists of 50,000 movie reviews, split evenly into 25k for training and 25k for testing. Each review is labeled as either positive or negative.

## Installation
To run the project locally, follow these steps:

1. Clone this repository:
```
git clone https://github.com/VuBacktracking/mamba-text-classification.git
```
2. Install the required dependencies:
```
pip install -r requirements.txt
```

## Usage
1. Navigate to the project directory:
```
cd mamba-text-classification
python trainer.py
```

## History of my training
| Step | Training Loss | Validation Loss | Accuracy |
|------|---------------|-----------------|----------|
| 625  | 0.020500      | 0.246246        | 0.928000 |
| 1250 | 0.671000      | 0.195849        | 0.940800 |
| 1875 | 0.596100      | 0.266093        | 0.934400 |
| 2500 | 0.016700      | 0.217099        | 0.941200 |
| 3125 | 0.000700      | 0.209536        | 0.944800 |
| 3750 | 2.680700      | 0.188751        | 0.949200 |
| 4375 | 0.015500      | 0.224948        | 0.950000 |
| 5000 | 0.002100      | 0.199092        | 0.952800 |
| 5625 | 0.013400      | 0.192042        | 0.952400 |
| 6250 | 0.152500      | 0.190083        | 0.953600 |

**Note**: You can check my model on hugging face hub in the link: https://huggingface.co/vubacktracking/mamba_text_classification

## Dependencies
- Python 3.x
- Other dependencies listed in requirements.txt

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- The IMDB dataset: [http://ai.stanford.edu/~amaas/data/sentiment/](http://ai.stanford.edu/~amaas/data/sentiment/)
- Mamba: [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/pdf/2312.00752)