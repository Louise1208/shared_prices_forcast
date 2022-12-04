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


# Standard the data
def data_standardization(df):
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
    x_train = train.drop('Close', axis=1)
    y_train = train['Close']
    x_valid = test.drop('Close', axis=1)

    return x_train, y_train, x_valid, train, test


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


def evaluation(train, valid, preds):
    y_valid = valid['Close']
    rmse = np.sqrt(np.mean(np.power((np.array(y_valid) - np.array(preds)), 2)))
    # TODO:accuracy 加几个指标

    # Plot the result of prediction
    valid['Predictions'] = 0
    valid['Predictions'] = preds

    plt.plot(valid[['Close', 'Predictions']])
    plt.plot(train['Close'])
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


def lstm_dataprocessing(df):
    # 筛选四个变量，作为数据的输入特征
    sel_col = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    df_LSTM = df[sel_col]

    # 归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    for col in sel_col:  # 这里不能进行统一进行缩放，因为fit_transform返回值是numpy类型
        max_min_scaler = scaler.fit_transform(df_LSTM[col].values.reshape(-1, 1))
        df_LSTM = df_LSTM.drop(col, axis=1)
        df_LSTM[col] = max_min_scaler
    # print(df_LSTM.head())

    # 将下一日的收盘价作为本日的标签
    df_LSTM['target'] = df_LSTM['Close'].shift(-1)
    # print(np.sum(df_LSTM.isnull()))
    df_LSTM.dropna()  # 使用了shift函数有缺失值，这里去掉缺失值所在行
    df_LSTM = df_LSTM.astype(np.float32)  # 修改数据类型
    return df_LSTM


if __name__ == '__main__':
    name = 'Tesla-dataset.csv'
    df = pd.read_csv(name)

    # 查看数据格式及数据信息
    df.info()
    df.describe()
    # 查看日期范围
    df['Date'].max()
    df['Date'].min()

    # 检查数据是否存在质量问题
    # 查看是否有缺失值
    print('缺失值有几个：\n', np.sum(df.isnull()))
    # 若有drop N/A, e.g. weekends, holidays
    df.dropna(how='any', inplace=True)
    # 对日期格式进行处理
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    # 查看是否存在非工作日数据：
    # 筛选工作日数据
    df['Date_working'] = df['Date'].dt.to_period('B')
    if len(df['Date_working'].unique()) == len(df['Date'].unique()):
        print('没有非工作日数据')
    # 删除新增的测试数据
    df.drop('Date_working', axis=1, inplace=True)  # drop函数默认删除行，列需要加axis = 1 ,inplace =true则直接覆盖原数组

    # 进行数据处理并寻找联系，依据图像等
    plt.title(label='the Details in Tesla stock', loc='center')
    plt.xlabel('Date')
    plt.ylabel('Price')
    # 根据以上结果调整刻度大小
    ax = plt.gca()  # 表明设置图片的各个轴
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # 横坐标标签显示的日期格式
    ax.xaxis.set_major_locator(mdates.YearLocator())  # 以月为定为符
    plt.yticks(range(2, 1300, 300))  # 设置纵坐标，使用range()函数设置起始、结束范围及间隔步长
    # 加入数据
    x = df['Date']
    plt.plot(x, df['Open'], label='Open')
    plt.plot(x, df['Close'], label='Close')
    plt.plot(x, df['High'], label='High')
    plt.plot(x, df['Low'], label='Low')
    plt.legend(loc='best')
    plt.show()
    # reset index after drop
    df.reset_index(inplace=True)

    # TODO:reudce the data size and 作图
    # 若有drop N/A, e.g. weekends, holidays
    # df.dropna(how='any', inplace=True)
    # reset index after drop
    df.reset_index(inplace=True)
    # setting index as date
    df['Date'] = pd.to_datetime(df.Date, format='%Y-%m-%d')
    print(len(df['Date']))
    pd['Date'].date_range(freq='b')
    print(len(df['Date']))

    # show the volumn and close price
    df.plot(x='Volume', y='Close', kind='scatter')
    plt.title(label='Relationship between Volume and Close', loc='center')
    plt.xlabel('Volume')
    plt.ylabel('Close')
    plt.show()

    # show the Close Price history
    plt.title(label='close price history', loc='center')
    plt.plot(df['Date'], df['Close'], label='Close', color='red')
    plt.xlabel('date')
    plt.ylabel('Close')
    plt.show()

    df.index = df['Date']
    df.drop('index', axis=1, inplace=True)
    date_length = len(df)
    # rmse 很低。可能：tesla是最近几年崛起的，因此使用全不数据意义不大。可以只使用最近几年的数据
    df = df[len(df) - 1500:].copy()
    # start to predict
    x_train, y_train, x_test, train, test = data_standardization(df)

    # TODO： knn和线性回归，修改，增加一个成交量与收盘价格的关系的预测！！
    # using knn
    preds = knn_model(x_train, y_train, x_test)
    knn_rmse = evaluation(train, test, preds)
    print('the RMSE of knn module is ', knn_rmse)

    # using linear regression
    preds = linear_model(x_train, y_train, x_test)
    linear_rmse = evaluation(train, test, preds)
    print('the RMSE of linear regression module is ', linear_rmse)

    # TODO: LSTM MODIFY
    # using lstm
    df_main = lstm_dataprocessing(df)
    # 创建两个列表，用来存储数据的特征和标签
    data_feat, data_target = [], []

    # 设每条数据序列有20组数据
    seq = 20
    for index in range(len(df_main) - seq):
        # 构建特征集
        data_feat.append(df_main[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']][index: index + seq].values)
        # 构建target集
        data_target.append(df_main['target'][index:index + seq])

    # 将特征集和标签集整理成numpy数组
    data_feat = np.array(data_feat)
    data_target = np.array(data_target)

    # 这里按照8:2的比例划分训练集和测试集
    test_set_size = int(np.round(0.2 * df_main.shape[0]))  # np.round(1)是四舍五入，
    train_size = data_feat.shape[0] - (test_set_size)
    print(test_set_size)  # 输出测试集大小
    print(train_size)  # 输出训练集大小

    # tensor
    trainX = torch.from_numpy(data_feat[:train_size].reshape(-1, seq, 6)).type(torch.Tensor)
    testX = torch.from_numpy(data_feat[train_size:].reshape(-1, seq, 6)).type(torch.Tensor)
    trainY = torch.from_numpy(data_target[:train_size].reshape(-1, seq, 1)).type(torch.Tensor)
    testY = torch.from_numpy(data_target[train_size:].reshape(-1, seq, 1)).type(torch.Tensor)
    print('x_train.shape = ', trainX.shape)
    print('y_train.shape = ', trainY.shape)
    print('x_test.shape = ', testX.shape)
    print('y_test.shape = ', testY.shape)

    input_dim = 6  # 数据的特征数
    hidden_dim = 16  # 隐藏层的神经元个数
    num_layers = 2  # LSTM的层数
    output_dim = 1  # 预测值的特征数
    # （这是预测股票价格，所以这里特征数是1，如果预测一个单词，那么这里是one-hot向量的编码长度）

    # 实例化模型
    model = LSTM(_input_dim=input_dim, _hidden_dim=hidden_dim, _output_dim=output_dim, _hidden_layer=num_layers)

    # 定义优化器和损失函数
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)  # 使用Adam优化算法
    loss_fn = torch.nn.MSELoss(reduction='mean')  # 使用均方差作为损失函数

    # 设定数据遍历次数
    num_epochs = 100

    # 打印模型结构
    print(model)

    # 打印模型各层的参数尺寸
    for i in range(len(list(model.parameters()))):
        print(list(model.parameters())[i].size())
    # 分块
    batch_size = 500
    train = torch.utils.data.TensorDataset(trainX, trainY)
    test = torch.utils.data.TensorDataset(testX, testY)
    train_loader = torch.utils.data.DataLoader(dataset=train,
                                               batch_size=batch_size,
                                               shuffle=False)

    test_loader = torch.utils.data.DataLoader(dataset=test,
                                              batch_size=batch_size,
                                              shuffle=False)
    # train model
    hist = np.zeros(num_epochs)
    for t in range(num_epochs):
        for step, (x, y) in enumerate(train_loader):

            # Initialise hidden state
            # Don't do this if you want your LSTM to be stateful
            # model.hidden = model.init_hidden()
            # Forward pass
            y_train_pred = model.forward(x)

            loss = loss_fn(y_train_pred, y)
            if t % 10 == 0 and t != 0:  # 每训练十次，打印一次均方差
                print("Epoch ", t, "MSE: ", loss.item())
            hist[t] = loss.item()

            # Zero out gradient, else they will accumulate between epochs 将梯度归零
            optimiser.zero_grad()
            # Backward pass
            loss.backward()
            # Update parameters
            optimiser.step()

    # 计算训练得到的模型在训练集上的均方差
    y_train_pred = model.forward(trainX)
    print(loss_fn(y_train_pred, trainY).item())

    # 作图
    # TODO 做细节图
    # 无论是真实值，还是模型的输出值，它们的维度均为（batch_size, seq, 1），seq=20
    # 我们的目的是用前20天的数据预测今天的股价，所以我们只需要每个数据序列中第20天的标签即可
    # 因为前面用了使用DataFrame中shift方法，所以第20天的标签，实际上就是第21天的股价
    pred_value = y_train_pred.detach().numpy()[:, -1, 0]
    true_value = trainY.detach().numpy()[:, -1, 0]
    plt.plot(pred_value, label="Preds")  # 预测值
    plt.plot(true_value, label="Data")  # 真实值
    plt.legend()
    plt.show()

# TODO
