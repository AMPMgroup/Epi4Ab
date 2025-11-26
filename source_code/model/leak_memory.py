#https://discuss.pytorch.org/t/memory-leak-debugging-and-common-causes/67339/14

import torch

class LeakFinder:

    def __init__(self):
        self.step = 0  # used to keep track of the step in the batch
        self.batch = 0  # used to keep track of the batch
        self.values = {}
        self.predict_every = 20  # how often to predict the leak position
        self.verbose = True  # print the predicted leak position

    def set_batch(self, epoch):
        """
        Set the batch number
        """
        self.batch = epoch
        self.step = 0
        self.values[epoch] = {}

    def get_cuda_perc(self):

        # get the percentage of cuda memory used
        perc = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
        self.values[self.batch][self.step] = perc * 100

        self.step += 1

    def predict_leak_position(self, diffs, per_epoch_remainder):
        # train a tree regressor to predict the per epoch increase
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error
        from sklearn.preprocessing import MinMaxScaler

        # insert a zero at the start of  per_epoch_remainder
        per_epoch_remainder = torch.cat([torch.tensor([0]), per_epoch_remainder])

        # scale the data to be between 0 and 1
        x_scaler = MinMaxScaler()
        diffs = x_scaler.fit_transform(diffs)

        y_scaler = MinMaxScaler()
        per_epoch_remainder = y_scaler.fit_transform(per_epoch_remainder.reshape(-1, 1))

        # train test split
        X_train, X_test, y_train, y_test = train_test_split(diffs, per_epoch_remainder, test_size=0.1, random_state=42)

        # train regressor
        regressor = DecisionTreeRegressor(random_state=0)
        regressor.fit(X_train, y_train)

        # predict
        y_pred = regressor.predict(X_test)

        # calculate error
        mse = mean_squared_error(y_test, y_pred)
        mag = mse / per_epoch_remainder.mean() * 100
        print(f"MSE: {mse} ({mag:.2f}%)")

        # find the most important feature
        feature_importance = regressor.feature_importances_
        most_important_feature = torch.argmax(torch.tensor(feature_importance))
        print(f"Likely leak position between step {most_important_feature} and step {most_important_feature + 1}")

    def find_leaks(self):
        """
        Find leaks in the training loop
        """

        if self.batch < 2:
            return

        if not self.verbose and self.batch % self.predict_every != 0:
            return

        # estimate per step diff
        diffs = []
        for epoch, values in self.values.items():
            dif = []
            for step in range(1, len(values)):
                dif += [values[step] - values[step - 1]]
            diffs.append(dif)

        lens = [len(x) for x in diffs]
        min_lens = min(lens)

        per_epoch_increase = [self.values[epoch][min_lens - 1] - self.values[epoch][0] for epoch in self.values.keys()
                              if epoch > 0]
        between_epoch_decrease = [self.values[epoch][0] - self.values[epoch - 1][min_lens - 1] for epoch in
                                  self.values.keys() if epoch > 0]
        per_epoch_increase = torch.tensor(per_epoch_increase)
        between_epoch_decrease = torch.tensor(between_epoch_decrease)

        per_epoch_remainder = per_epoch_increase + between_epoch_decrease

        per_epoch_increase_mean = per_epoch_remainder.mean()
        per_epoch_increase_sum = per_epoch_remainder.sum()

        diffs = torch.tensor(diffs)

        print(
            f"Per epoch increase: {per_epoch_increase_mean:.2f}% cuda memory "
            f"(total increase of {per_epoch_increase_sum:.2f}%) currently at "
            f"{self.values[self.batch][min_lens - 1]:.2f}% cuda memory")

        if self.batch % self.predict_every == 0:
            self.predict_leak_position(diffs, per_epoch_remainder)

def print_cuda_memory(message = ''):
    max = torch.cuda.max_memory_allocated()
    if max == 0:
        print(f'{message}: Max memory is 0')
    else:
        perc = torch.cuda.memory_allocated() / max
        print(f'{message}: Memory Percent: {perc:.2f}%')