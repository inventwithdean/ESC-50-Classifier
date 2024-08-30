# Audio Classification with ESC-50 Dataset

This repository contains my work on building and training a deep convolutional neural network (CNN) for audio classification using the [ESC-50 dataset](https://github.com/karolpiczak/ESC-50). The ESC-50 dataset is a labeled collection of 2000 environmental audio recordings, suitable for benchmarking methods of environmental sound classification.

## Project Overview

- **Objective:** To develop a deep learning model that classifies audio clips into 50 different categories using spectrogram representations.
- **Dataset:** [ESC-50: Dataset for Environmental Sound Classification](https://github.com/karolpiczak/ESC-50)
- **Model:** A ResNet-inspired CNN architecture to classify spectrogram images of audio signals.
- **Tools & Libraries:** Python, PyTorch, torchaudio, pandas.

## Key Steps

1. **Data Preparation:**
   - Loaded and preprocessed audio data from the ESC-50 dataset.
   - Converted audio signals to mel-spectrograms for use with a CNN.

2. **Model Architecture:**
   - Used a ResNet-inspired CNN architecture designed for image-based classification.
   - Fine-tuned the architecture to optimize performance on spectrogram data via cross-validation.

4. **Training:**
   - Trained the model using 4 out of 5 folds of the dataset, reserving the 5th fold for validation.
   - Achieved a validation accuracy of **65.50%**.

5. **Evaluation:**
   - Evaluated the model performance on unseen validation data.
   - Mapped model output classes to actual class names for interpretability.

## Results

- **Training Accuracy:** 99.94%
- **Validation Accuracy:** 65.50%

## Future Improvements

- Experimenting with more complex architectures such as EfficientNet or ConvNeXt.
- Fine-tuning the model with more advanced augmentation techniques.
- Further hyperparameter optimization and regularization to reduce overfitting.

## Usage

To replicate the results, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/inventwithdean/ESC-50-classifier.git
   ```
3. Run the Notebook:
   ```bash
   jupyter notebook AudioDL.py
   ```

## Acknowledgments

- Thanks to [Karol J. Piczak](https://github.com/karolpiczak) for the ESC-50 dataset.
- Inspiration from various deep learning resources and the PyTorch community.

## License

This project is licensed under the MIT License.

## Contributing

Feel free to submit issues or pull requests for any improvements or bug fixes!
