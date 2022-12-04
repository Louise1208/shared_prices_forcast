import matplotlib.dates as mdates  # read file and store in dataframes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data
from fastai.tabular.all import add_datepart
from sklearn import neighbors
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler


def data_preparation(df):
    standardized_data = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Close'])
    for i in range(0, len(df)):
        standardized_data['Date'][i] = df['Date'][i]
        standardized_data['Close'][i] = df['Close'][i]
    # Add all the features of date
    add_datepart(standardized_data, 'Date')
    # Drop Elapsed because model do not accept 'Timestamp' type
    standardized_data.drop('Elapsed', axis=1, inplace=True)
    # print(data_close.head())
    # Split data into train, test and validation
    split_point = int(len(standardized_data) * 0.8)
    train = standardized_data[:split_point]
    test = standardized_data[split_point:]
    train_x = train.drop('Close', axis=1)
    train_y = train['Close']
    test_x = test.drop('Close', axis=1)
    test_y = test['Close']

    return train_x, train_y, test_x, test_y


def knn_model(x_train, y_train, x_test):
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Scale data
    x_train = pd.DataFrame(scaler.fit_transform(x_train))
    x_test = pd.DataFrame(scaler.fit_transform(x_test))
    # Find the best parameter by grid search
    params = {'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]}
    knn = neighbors.KNeighborsRegressor()
    model = GridSearchCV(knn, params, cv=5)
    model.fit(x_train, y_train)
    print("best parameters: ", model.best_params_)
    # Predict test data
    preds = model.predict(x_test)

    return preds


def linear_model(x_train, y_train, x_test):
    model = LinearRegression()
    model.fit(x_train, y_train)
    preds = model.predict(x_test)

    return preds


def evaluation(test_y, preds):
    rmse = np.sqrt(np.mean(np.power((np.array(test_y) - np.array(preds)), 2)))
    plt.plot(test_y, label='Actual Data')
    plt.plot(test_y.index, preds, label='Predicted Data')
    plt.show()

    return rmse


# TODO: 用LSTM模型
class LSTM(nn.Module):
    def __init__(self, _input_dim, _hidden_dim, _hidden_layer, _output_dim):
        super(LSTM, self).__init__()
        self.input_dim = _input_dim
        self.hidden_dim = _hidden_dim
        self.num_layers = _hidden_layer
        # Build LSTM
        # batch_first=True causes input/output tensors to be of shape (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(_input_dim, _hidden_dim, _hidden_layer, batch_first=True)
        # Add a full connected layer
        self.fc = nn.Linear(_hidden_dim, _output_dim)

    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        # Detach them because we don't want to propagate back through the entire training history
        h0 = h0.detach()
        c0 = c0.detach()
        # One time step
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out)

        return out


def lstm_data_preparation(df):
    sel_col = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    df_LSTM = df[sel_col]
    # Normalisation
    scaler = MinMaxScaler(feature_range=(0, 1))
    for col in sel_col:
        max_min_scaler = scaler.fit_transform(df_LSTM[col].values.reshape(-1, 1))
        df_LSTM = df_LSTM.drop(col, axis=1)
        df_LSTM[col] = max_min_scaler
    # print(df_LSTM.head())
    # Use the next close price as today's target
    df_LSTM['target'] = df_LSTM['Close'].shift(-1)
    # Drop data because of shift function
    df_LSTM.dropna()
    # Create the feature set and target set
    data_features, data_targets = [], []
    seq = 20
    for index in range(len(df_LSTM) - seq):
        data_features.append(
            df_LSTM[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']][index: index + seq].values)
        data_targets.append(df_LSTM['target'][index:index + seq])
    data_features = np.array(data_features)
    data_targets = np.array(data_targets)
    # Split the data into train and test according to the ratio
    ratio = 0.2
    test_set_size = int(np.round(ratio * df_LSTM.shape[0]))
    train_size = data_features.shape[0] - test_set_size
    train_x = torch.from_numpy(data_features[:train_size].reshape(-1, seq, 6)).type(torch.Tensor)
    test_x = torch.from_numpy(data_features[train_size:].reshape(-1, seq, 6)).type(torch.Tensor)
    train_y = torch.from_numpy(data_targets[:train_size].reshape(-1, seq, 1)).type(torch.Tensor)
    test_y = torch.from_numpy(data_targets[train_size:].reshape(-1, seq, 1)).type(torch.Tensor)

    return train_x, train_y, test_x, test_y


# TODO Pre-analysis the data
def data_pre_analysis(df):
    plt.title(label='the Details in Tesla stock', loc='center')
    plt.xlabel('Date')
    plt.ylabel('Price')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.yticks(range(2, 1300, 300))
    # 加入数据
    x = df['Date']
    plt.plot(x, df['Open'], label='Open')
    plt.plot(x, df['Close'], label='Close')
    plt.plot(x, df['High'], label='High')
    plt.plot(x, df['Low'], label='Low')
    plt.legend(loc='best')
    plt.show()

    df['Date'] = pd.to_datetime(df.Date, format='%Y-%m-%d')
    df.plot(x='Volume', y='Close', kind='scatter')
    plt.title(label='Relationship between Volume and Close', loc='center')
    plt.xlabel('Volume')
    plt.ylabel('Close')
    plt.show()
    # Plot the Close Price history
    plt.title(label='close price history', loc='center')
    plt.plot(df['Date'], df['Close'], label='Close', color='red')
    plt.xlabel('date')
    plt.ylabel('Close')
    plt.show()


if __name__ == '__main__':
    file_path = 'Tesla-dataset.csv'
    df = pd.read_csv(file_path)

    # Describe the data
    df.info()
    df.describe()
    # Process the data
    df.dropna(how='any', inplace=True)
    # Transform the data type
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    df.reset_index(inplace=True)
    df.index = df['Date']
    df.drop('index', axis=1, inplace=True)
    data_pre_analysis(df)
    # TODO rmse 很低。可能：tesla是最近几年崛起的，因此使用全不数据意义不大。可以只使用最近几年的数据
    df = df[len(df) - 1500:].copy()

    x_train, train_y, x_test, test_y = data_preparation(df)
    # Use KNN model
    preds = knn_model(x_train, train_y, x_test)
    knn_rmse = evaluation(test_y, preds)
    print('The RMSE of knn module: ', knn_rmse)
    # Use linear regression Model
    preds = linear_model(x_train, train_y, x_test)
    linear_rmse = evaluation(test_y, preds)
    print('The RMSE of linear regression model: ', linear_rmse)
    # Use LSTM model
    train_x, train_y, test_x, test_y = lstm_data_preparation(df)
    input_dim = 6
    hidden_dim = 16
    hidden_layer = 2
    output_dim = 1
    model = LSTM(_input_dim=input_dim, _hidden_dim=hidden_dim, _output_dim=output_dim, _hidden_layer=hidden_layer)
    # Create the optimizer by Adam
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
    # Create the loss function by MSE
    loss_fn = torch.nn.MSELoss(reduction='mean')

    epoch_num = 100
    batch_size = 500
    train = torch.utils.data.TensorDataset(train_x, train_y)
    test = torch.utils.data.TensorDataset(test_x, test_y)
    train_loader = torch.utils.data.DataLoader(dataset=train,
                                               batch_size=batch_size,
                                               shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test,
                                              batch_size=batch_size,
                                              shuffle=False)
    # Train model
    hist = np.zeros(epoch_num)
    for t in range(epoch_num):
        for step, (x, y) in enumerate(train_loader):
            preds = model.forward(x)
            loss = loss_fn(preds, y)
            if t % 10 == 0 and t != 0:
                print("Epoch ", t, "MSE: ", loss.item())
            hist[t] = loss.item()
            # Zero out gradient, else they will accumulate between epochs
            optimiser.zero_grad()
            # Backward pass
            loss.backward()
            # Update parameters
            optimiser.step()

    preds = model.forward(train_x)
    print("Loss: ", loss_fn(preds, train_y).item())

    # TODO 做细节图
    # 无论是真实值，还是模型的输出值，它们的维度均为（batch_size, seq, 1），seq=20
    # 我们的目的是用前20天的数据预测今天的股价，所以我们只需要每个数据序列中第20天的标签即可
    # 因为前面用了使用DataFrame中shift方法，所以第20天的标签，实际上就是第21天的股价
    # Plot the prediction
    pred_value = preds.detach().numpy()[:, -1, 0]
    true_value = train_y.detach().numpy()[:, -1, 0]
    plt.plot(pred_value, label="Preds")
    plt.plot(true_value, label="Data")
    plt.legend()
    plt.show()
