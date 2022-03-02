import src.base as base
import src.functional as F


class Fscore(base.Metric):

    def __init__(self, beta=1, eps=1e-7,
                 onehot=False, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.threshold = threshold
        self.activation = base.Activation(activation)
        self.ignore_channels = ignore_channels
        self.onehot = onehot

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.f_score(
            y_pr, y_gt,
            onehot=self.onehot,
            eps=self.eps,
            beta=self.beta,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class Accuracy(base.Metric):

    def __init__(self, threshold=0.5, activation=None,
                 onehot=False, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.activation = base.Activation(activation)
        self.ignore_channels = ignore_channels
        self.onehot = onehot

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.accuracy(
            y_pr, y_gt,
            onehot=self.onehot,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class Recall(base.Metric):

    def __init__(self, eps=1e-7,
                 onehot=False,
                 threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.onehot = onehot
        self.activation = base.Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.recall(
            y_pr, y_gt,
            onehot=self.onehot,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class Precision(base.Metric):

    def __init__(self, eps=1e-7,
                 onehot=False,
                 threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.onehot = onehot
        self.threshold = threshold
        self.activation = base.Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.precision(
            y_pr, y_gt,
            onehot=self.onehot,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class ConfusionMatrix(base.Metric):

    def __init__(self, threshold=0.5, activation=None,
                 onehot=False, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.activation = base.Activation(activation)
        self.ignore_channels = ignore_channels
        self.onehot = onehot

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.confusionmatrix(
            y_pr, y_gt,
            onehot=self.onehot,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )
