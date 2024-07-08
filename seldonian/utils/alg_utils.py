import torch
from torch import nn
from torch.distributions import Categorical

def update_adversary(model, batch_features):
    if hasattr(model, 'discriminator'):
        model.pytorch_model.eval()
        model.vfae.eval()
        if type(batch_features) == list:
            X, S, Y = batch_features
            X_torch = torch.tensor(X).float().to(model.device, non_blocking=True)
            S_torch = torch.tensor(S).float().to(model.device, non_blocking=True)
            Y_torch = torch.tensor(Y).float().to(model.device, non_blocking=True)
            model.pytorch_model(X_torch, S_torch, Y_torch, model.discriminator)
        else:
            X_torch = torch.tensor(batch_features).float().to(model.device, non_blocking=True)
            model.pytorch_model(X_torch, model.discriminator)
        model.discriminator.train()
        model.optimizer_d.zero_grad()
        s_decoded = model.discriminator(model.pytorch_model.z)
        if model.pytorch_model.s_dim == 1:
            loss = nn.BCELoss()
            discriminator_loss = loss(s_decoded, model.pytorch_model.s)
        else:
            p_adversarial = Categorical(probs=s_decoded)
            log_p_adv = p_adversarial.log_prob(model.pytorch_model.s)
            discriminator_loss = -log_p_adv.mean(dim=0)
        # print(discriminator_loss)
        discriminator_loss.backward()
        model.optimizer_d.step()
        model.discriminator.eval()
        model.vfae.train()
        model.pytorch_model.train()

def train_downstream(model, X_train, Y_train, batch_size,
                     num_epochs, lr, z_dim, hidden_dim, y_dim, device):
    print("Training downstream model...")
    loss_list = []
    accuracy_list = []
    iter_list = []
    x_train_tensor = torch.from_numpy(X_train)
    y_train_label = torch.from_numpy(Y_train)
    train = torch.utils.data.TensorDataset(x_train_tensor, y_train_label)
    trainloader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True
    )
    activation = nn.ReLU()
    criterion = nn.BCELoss()
    model.pytorch_model.eval()
    model.vfae.eval()
    if hasattr(model, 'discriminator'):
        model.discriminator.eval()
    downstream_model = DecoderMLP(z_dim, hidden_dim, 1, activation).to(device) # model.vfae.decoder_y
    # downstream_model = model.vfae.decoder_y
    print(
        f"Running downstream gradient descent with batch_size: {batch_size}, num_epochs={num_epochs}"
    )
    itot = 0
    optimizer = torch.optim.Adam(downstream_model.parameters(), lr=lr)
    downstream_model.train()
    for epoch in range(num_epochs):
        for i, (features, labels) in enumerate(trainloader):
            # Load images
            features = features.float().to(device)
            labels = labels.to(device)

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()
            # get representations
            representations = model.get_representations(features)
            # get prediction
            y_pred = downstream_model.forward(representations)
            # get loss
            loss = criterion(y_pred, labels.float().unsqueeze(1))
            # loss backward
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                it = f"{i+1}/{len(trainloader)}"
                print(f"Epoch, it, itot, loss: {epoch},{it},{itot},{loss}")
            itot += 1
    downstream_model.eval()
    return downstream_model


class DecoderMLP(nn.Module):
    """
     Single hidden layer MLP used for decoding.
    """

    def __init__(self, in_features, hidden_dim, latent_dim, activation):
        super().__init__()
        self.lin_encoder = nn.Linear(in_features, hidden_dim)
        self.activation = activation
        # self.lin_encoder_2 = nn.Linear(hidden_dim, hidden_dim//2)
        # self.activation = activation
        # self.lin_out = nn.Linear(hidden_dim//2, latent_dim)
        self.lin_out = nn.Linear(hidden_dim, latent_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x = self.activation(self.lin_encoder(inputs))
        # x = self.activation(self.lin_encoder_2(x))
        return self.sigmoid(self.lin_out(x))