import torch.nn as nn
import torch.nn.functional as F
import torch


class CapsuleSequenceToGraph(nn.Module):
    def __init__(self, text_dim, audio_dim, video_dim, capsule_dim, capsule_num, routing, multi_dim, T_t, T_a, T_v):
        super(CapsuleSequenceToGraph, self).__init__()
        self.d_c = capsule_dim
        self.n = capsule_num
        self.routing = routing
        self.multi_dim = multi_dim
        # self.pc_dropout = dropout
        # create primary capsule
        self.W_tpc = nn.Parameter(torch.Tensor(T_t, self.n, text_dim, self.d_c))
        self.W_apc = nn.Parameter(torch.Tensor(T_a, self.n, audio_dim, self.d_c))
        self.W_vpc = nn.Parameter(torch.Tensor(T_v, self.n, video_dim, self.d_c))
        nn.init.xavier_normal_(self.W_tpc)
        nn.init.xavier_normal_(self.W_apc)
        nn.init.xavier_normal_(self.W_vpc)

        # create adjacent matrix by self-attention
        self.WQt = nn.Parameter(torch.Tensor(self.d_c, self.d_c))
        self.WKt = nn.Parameter(torch.Tensor(self.d_c, self.d_c))
        self.WQa = nn.Parameter(torch.Tensor(self.d_c, self.d_c))
        self.WKa = nn.Parameter(torch.Tensor(self.d_c, self.d_c))
        self.WQv = nn.Parameter(torch.Tensor(self.d_c, self.d_c))
        self.WKv = nn.Parameter(torch.Tensor(self.d_c, self.d_c))
        self.WQm = nn.Parameter(torch.Tensor(self.multi_dim, self.multi_dim))
        self.WKm = nn.Parameter(torch.Tensor(self.multi_dim, self.multi_dim))
        nn.init.xavier_normal_(self.WQt)
        nn.init.xavier_normal_(self.WQa)
        nn.init.xavier_normal_(self.WQv)
        nn.init.xavier_normal_(self.WKt)
        nn.init.xavier_normal_(self.WKa)
        nn.init.xavier_normal_(self.WKv)
        nn.init.xavier_normal_(self.WQm)
        nn.init.xavier_normal_(self.WKm)

        self.fc_t = nn.Linear(self.d_c, self.multi_dim)
        self.fc_a = nn.Linear(self.d_c, self.multi_dim)
        self.fc_v = nn.Linear(self.d_c, self.multi_dim)

    def forward(self, text, audio, video, batch_size):
        # get dimensionality
        T_t = text.shape[0]
        T_a = audio.shape[0]
        T_v = video.shape[0]
        # create primary capsule
        text_pri_caps = (torch.einsum('tbj, tnjd->tbnd', text, self.W_tpc)).permute(1, 0, 2, 3)
        audio_pri_caps = (torch.einsum('tbj, tnjd->tbnd', audio, self.W_apc)).permute(1, 0, 2, 3)
        video_pri_caps = (torch.einsum('tbj, tnjd->tbnd', video, self.W_vpc)).permute(1, 0, 2, 3)

        # routing mechanism does not participate in back propagation
        text_pri_caps_temp = text_pri_caps.detach()
        audio_pri_caps_temp = audio_pri_caps.detach()
        video_pri_caps_temp = video_pri_caps.detach()

        # begin routing
        for r in range(self.routing+1):
            if r == 0:
                b_t = torch.zeros(batch_size, T_t, self.n).cuda()  # initialize routing coefficients
                b_a = torch.zeros(batch_size, T_a, self.n).cuda()
                b_v = torch.zeros(batch_size, T_v, self.n).cuda()
            rc_t = F.softmax(b_t, 2)
            rc_a = F.softmax(b_a, 2)
            rc_v = F.softmax(b_v, 2)

            text_vertex = torch.tanh(torch.sum(text_pri_caps_temp * rc_t.unsqueeze(-1), 1))
            audio_vertex = torch.tanh(torch.sum(audio_pri_caps_temp * rc_a.unsqueeze(-1), 1))
            video_vertex = torch.tanh(torch.sum(video_pri_caps_temp * rc_v.unsqueeze(-1), 1))

            # update routing coefficients
            if r < self.routing:
                last = b_t
                new = ((text_vertex.unsqueeze(1)) * text_pri_caps_temp).sum(3)
                b_t = last + new

                last = b_a
                new = (audio_vertex.unsqueeze(1) * audio_pri_caps_temp).sum(3)
                b_a = last + new

                last = b_v
                new = (video_vertex.unsqueeze(1) * video_pri_caps_temp).sum(3)
                b_v = last + new

        # create vertex using the routing coefficients in final round
        text_vertex = torch.tanh(torch.sum(text_pri_caps * rc_t.unsqueeze(-1), 1))
        audio_vertex = torch.tanh(torch.sum(audio_pri_caps * rc_a.unsqueeze(-1), 1))
        video_vertex = torch.tanh(torch.sum(video_pri_caps * rc_v.unsqueeze(-1), 1))

        multi_vertex = torch.cat((self.fc_t(text_vertex), self.fc_a(audio_vertex), self.fc_v(video_vertex)), dim=1)

        # use self-attention to create adjacent matrix
        Q = torch.matmul(text_vertex, self.WQt)
        K = torch.matmul(text_vertex, self.WKt)
        adj_t = torch.eye(self.n).cuda() + F.relu(torch.bmm(Q, K.permute(0, 2, 1)) / self.d_c)
        Q = torch.matmul(audio_vertex, self.WQa)
        K = torch.matmul(audio_vertex, self.WKa)
        adj_a = torch.eye(self.n).cuda() + F.relu(torch.bmm(Q, K.permute(0, 2, 1)) / self.d_c)
        Q = torch.matmul(video_vertex, self.WQv)
        K = torch.matmul(video_vertex, self.WKv)
        adj_v = torch.eye(self.n).cuda() + F.relu(torch.bmm(Q, K.permute(0, 2, 1)) / self.d_c)
        Q = torch.matmul(multi_vertex, self.WQm)
        K = torch.matmul(multi_vertex, self.WKm)
        adj_m = torch.eye(3*self.n).cuda() + F.relu(torch.bmm(Q, K.permute(0, 2, 1)) / self.d_c)

        return text_vertex, audio_vertex, video_vertex, multi_vertex, adj_t, adj_a, adj_v, adj_m


if __name__ == "__main__":
    text = torch.randn(10, 8, 300)
    audio = torch.randn(30, 8, 74)
    video = torch.randn(20, 8, 35)
    batch_size = text.size(1)
    seq_dim = 18
    text_dim = text.size(2)
    audio_dim = audio.size(2)
    video_dim = video.size(2)
    T_t = text.size(0)
    T_a = audio.size(0)
    T_v = video.size(0)
    capsule_dim = 32
    capsule_num = 12
    routing = 3
    multi_dim = 16
    model = CapsuleSequenceToGraph(text_dim, audio_dim, video_dim, capsule_dim, capsule_num, routing, multi_dim, T_t, T_a, T_v)
    text_vertex, audio_vertex, video_vertex, multi_vertex, adj_t, adj_a, adj_v, adj_m = model(text, audio, video, batch_size)
    print(text_vertex.size(), audio_vertex.size(), video_vertex.size(), multi_vertex.size(), adj_t.size(), adj_a.size(), adj_v.size(), adj_m.size())