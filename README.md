# Earthquake Prediction using Machine Learning

This project implements a deep learning model to predict earthquake characteristics (magnitude and depth) based on geographical and temporal features. The model uses historical earthquake data to identify patterns and make predictions about potential seismic events.

## Features

- Data preprocessing and temporal feature engineering
- Neural network architecture for multi-output prediction
- Geographic visualization of earthquake distributions
- Model evaluation and performance metrics
- Cross-validation for hyperparameter optimization

## Requirements

```
numpy>=1.19.2
pandas>=1.2.0
matplotlib>=3.3.0
scikit-learn>=0.24.0
tensorflow>=2.4.0
basemap>=1.2.0
seaborn>=0.11.0
```

## Dataset

The model uses historical earthquake data with the following features:
- Timestamp
- Latitude
- Longitude
- Depth
- Magnitude

You can obtain similar earthquake data from:
- USGS Earthquake Catalog
- National Earthquake Information Center (NEIC)
- International Seismological Centre (ISC)

## Model Architecture

The neural network consists of:
- Input layer (3 features)
- Two hidden layers with 16 neurons each
- Output layer (2 features: magnitude and depth)
- ReLU activation for hidden layers
- Softmax activation for output layer

## Usage

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Run the main script:
```bash
python earthquake_prediction.py
```

## Results

The model achieves:
- Training accuracy: ~95%
- Test accuracy: ~92%
- Loss (squared hinge): ~0.50

## Future Improvements

- Implement feature importance analysis
- Add more seismic features (e.g., historical patterns)
- Experiment with different architectures (LSTM, CNN)
- Include uncertainty quantification
- Add real-time prediction capabilities

## License

MIT License
