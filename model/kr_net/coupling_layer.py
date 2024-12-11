import torch
import torch.nn as nn


class AffineCouplingLayer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim=24, num_modules=2, alternating=False):
        super(AffineCouplingLayer, self).__init__()
        self.isAlternating = alternating
        if self.isAlternating:
            self.num_identity = 1
            self.num_change = input_dim - 1
        else:
            self.num_identity = input_dim - 1
            self.num_change = 1
        self.num_split = max(input_dim - 1, 1)

        self.alpha = 0.6
        self.exp_beta = _RescaleT(self.num_change)
        self.tanh = nn.Tanh()

        self.st_net = _STNet(self.num_identity, self.num_change * 2, hidden_dim, num_modules)

    def forward(self, x, pdj=None, sldj=None, reverse=True):
        if self.isAlternating:
            x_change, x_id = x.split(self.num_split, dim=1)
        else:
            x_id, x_change = x.split(self.num_split, dim=1)

        st = self.st_net(x_id)
        s, t = st.chunk(2, dim=1)
        s = 1 + self.alpha * self.tanh(s)
        t = self.exp_beta(self.tanh(t))

        # Scale and translate
        if reverse:
            x_change = (x_change - t) / s

        # x_change = (x_change - t) * (-s).exp()
        else:
            x_change = x_change * s + t

            # x_change = x_change * s.exp() + t

            # Time abs-determinant of the Jacobian
            # Remark s > 0 always hold since s = 1 + \alpha * tanh(s) > 0 as long as \alpha < 1
            pdj = pdj * s.prod(-1, keepdims=True)

            # pdj = pdj * s.exp().prod(-1, keepdims=True)

            # Add log-abs-determinant of the Jacobian
            sldj = sldj + s.log().sum(-1, keepdims=True)

            # sldj = sldj + s.sum(-1, keepdims=True)

        if self.isAlternating:
            x = torch.cat((x_change, x_id), dim=1)
        else:
            x = torch.cat((x_id, x_change), dim=1)

        return x, pdj, sldj


class _RescaleT(nn.Module):
    def __init__(self, num_change):
        super(_RescaleT, self).__init__()
        self.weight = nn.Parameter(torch.zeros(num_change))

    def forward(self, x):
        x = self.weight.exp() * x
        return x


class _STNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim=48, num_modules=2):
        super(_STNet, self).__init__()
        self.in_module = LinearModule(input_dim, hidden_dim, 'relu')
        self.hidden_modules = nn.ModuleList(
            [LinearModule(hidden_dim, hidden_dim, 'relu') for _ in range(num_modules - 1)])
        self.out_module = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.in_module(x)
        for module in self.hidden_modules:
            x = module(x)
        x = self.out_module(x)
        return x


class LinearModule(nn.Module):
    """
    A combination of a linear layer followed with an activation function.
    """

    def __init__(self, data_dim, hidden_dim, activation_name, *args):
        super().__init__()

        acti = {
            "tanh": nn.Tanh(),
            "relu": nn.ReLU(),
            "leaky relu": nn.LeakyReLU(args),
        }
        assert activation_name in acti.keys(), f"{[act for act in acti.keys()]} are used while input is {activation_name}"

        self.linear = nn.Linear(data_dim, hidden_dim)

        self.activation = acti[activation_name]

        # init weight and bias
        nn.init.normal_(self.linear.weight, std=0.1)

    def forward(self, x):
        y = self.linear(x)
        y = self.activation(y)
        return y
