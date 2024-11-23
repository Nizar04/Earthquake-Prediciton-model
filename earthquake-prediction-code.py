import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import datetime
import time
import seaborn as sns

class EarthquakePrediction:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        
    def load_data(self, filepath):
        data = pd.read_csv(filepath)
        return data[['Date', 'Time', 'Latitude', 'Longitude', 'Depth', 'Magnitude']]
    
    def process_timestamps(self, data):
        timestamp = []
        for d, t in zip(data['Date'], data['Time']):
            try:
                ts = datetime.datetime.strptime(d+' '+t, '%m/%d/%Y %H:%M:%S')
                timestamp.append(time.mktime(ts.timetuple()))
            except ValueError:
                timestamp.append(np.nan)
        
        data['Timestamp'] = timestamp
        data = data.dropna()
        return data.drop(['Date', 'Time'], axis=1)
    
    def visualize_distribution(self, data):
        plt.figure(figsize=(12, 10))
        m = Basemap(projection='mill', llcrnrlat=-80, urcrnrlat=80, 
                   llcrnrlon=-180, urcrnrlon=180, lat_ts=20, resolution='c')
        
        x, y = m(data["Longitude"].tolist(), data["Latitude"].tolist())
        
        plt.title("Earthquake Distribution")
        m.plot(x, y, "o", markersize=2, color='blue', alpha=0.6)
        m.drawcoastlines()
        m.fillcontinents(color='lightgray', lake_color='aqua')
        m.drawmapboundary()
        m.drawcountries()
        plt.show()
    
    def prepare_data(self, data):
        X = data[['Timestamp', 'Latitude', 'Longitude']]
        y = data[['Magnitude', 'Depth']]
        
        X_scaled = self.scaler.fit_transform(X)
        
        return train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    def create_model(self, neurons=16, activation='relu'):
        model = Sequential([
            Dense(neurons, activation=activation, input_shape=(3,)),
            Dropout(0.2),
            Dense(neurons, activation=activation),
            Dropout(0.2),
            Dense(2, activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                     loss='squared_hinge',
                     metrics=['accuracy'])
        return model
    
    def train_model(self, X_train, y_train, X_test, y_test):
        model = KerasClassifier(build_fn=self.create_model, verbose=0)
        
        param_grid = {
            'neurons': [16, 32],
            'batch_size': [10, 20],
            'epochs': [10],
            'activation': ['relu', 'tanh']
        }
        
        grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1)
        grid_result = grid.fit(X_train, y_train)
        
        best_params = grid_result.best_params_
        self.model = self.create_model(
            neurons=best_params['neurons'],
            activation=best_params['activation']
        )
        
        history = self.model.fit(
            X_train, y_train,
            batch_size=best_params['batch_size'],
            epochs=best_params['epochs'],
            validation_data=(X_test, y_test)
        )
        
        return history, best_params
    
    def evaluate_model(self, X_test, y_test):
        test_loss, test_acc = self.model.evaluate(X_test, y_test)
        return test_loss, test_acc
    
    def predict(self, features):
        scaled_features = self.scaler.transform(features)
        return self.model.predict(scaled_features)

def main():
    predictor = EarthquakePrediction()
    
    data = predictor.load_data("database.csv")
    processed_data = predictor.process_timestamps(data)
    
    predictor.visualize_distribution(processed_data)
    
    X_train, X_test, y_train, y_test = predictor.prepare_data(processed_data)
    
    history, best_params = predictor.train_model(X_train, y_train, X_test, y_test)
    test_loss, test_acc = predictor.evaluate_model(X_test, y_test)
    
    print(f"Best parameters: {best_params}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")

if __name__ == "__main__":
    main()
