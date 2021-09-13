import torch

class LstmCell(torch.nn.Module):
  def __init__(self, state_dim, input_dim):
    super(LstmCell, self).__init__()
    self.linearx = torch.nn.Linear(input_dim, 4 * state_dim)
    self.linearh = torch.nn.Linear(state_dim, 4 * state_dim, bias=False)
    self.sigmoid = torch.nn.Sigmoid()
    self.tanh = torch.nn.Tanh()

  def forward(self, x, prevh, prevc):
    # x.shape: (N,D)
    # prevh.shape: (N,H)
    # prevc.shape: (N,H)
    stacked_output = self.linearx(x) + self.linearh(prevh)
    i_bar, f_bar, o_bar, g_bar = torch.chunk(stacked_output, 4, dim=1)
    i = self.sigmoid(i_bar)
    f = self.sigmoid(f_bar)
    o = self.sigmoid(o_bar)
    g = self.tanh(g_bar)
    nextc = (prevc * f) + (g * i)
    nexth = self.tanh(nextc) * o
    return nexth, nextc

