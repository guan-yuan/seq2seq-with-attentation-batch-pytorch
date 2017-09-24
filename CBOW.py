# this a CBOW implementation using Pytorch
# modified by http://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html

raw_text = """台北 世大運 19 日 開幕 為期 12 天 有 134 國 近萬
運動員 參賽 在 7 萬名 工作 人員 1 萬 多名 志工 參與 之下 舉辦 271 個
 比賽 項目 頒出 1978 面獎 牌 有 超過 72 萬 人次 的 觀眾 進場 觀賞 賽事 台灣
代表 團喊 出 回家 比賽 以 26 金 34 銀 30 銅 排名 第三 創造 史上 最佳 紀錄""".split()

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

CONTEXT_SIZE = 2  # windows size == 2 * CONTEXT_SIZE
EMBEDDING_DIM = 10


# By deriving a set from `raw_text`, we deduplicate the array
vocab = set(raw_text)
vocab_size = len(vocab)

word_to_ix = {word: i for i, word in enumerate(vocab)}
data = []
for i in range(2, len(raw_text) - 2):
    context = [raw_text[i - 2], raw_text[i - 1],
               raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]
    data.append((context, target))


class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).sum(dim=0).view((1, -1))
        out = self.linear1(embeds)
        log_probs = F.log_softmax(out)
        return log_probs


def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)


context = make_context_vector(data[0][0], word_to_ix)  # example

# create your model and train.  here are some functions to help you make
# the data ready for use by your module
losses = []
loss_function = nn.NLLLoss()
model = CBOW(vocab_size, EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(20000):
    total_loss = torch.Tensor([0])
    for context, target in data:

        context_idxs = [word_to_ix[w] for w in context]
        context_var = autograd.Variable(torch.LongTensor(context_idxs))

        model.zero_grad()

        log_probs = model(context_var)

        loss = loss_function(log_probs, autograd.Variable(
            torch.LongTensor([word_to_ix[target]])))


        loss.backward()
        optimizer.step()

        total_loss += loss.data
    losses.append(total_loss)
print(losses)


