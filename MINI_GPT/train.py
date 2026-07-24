from xml.parsers.expat import model

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-3
torch.device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

torch.manual_seed(1337)


# Load your dataset
loader = TextLoader("dataset_bank/Input.txt", encoding="utf-8")
documents = loader.load()

# Extract actual text from Document objects
text = "\n".join(doc.page_content for doc in documents)

chars = sorted(list(set(text)))
vocab_size = len(chars)
# print(''.join(chars))
# print(f"Vocabulary size: {vocab_size}")

# Create a mapping from string to char and char to string
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encoder = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
decoder = lambda l: ''.join([itos[i] for i in l])  # decoder

# print(encoder("hello there"))
# print(decoder(encoder("hello there")))

data = torch.tensor(encoder(text), dtype=torch.long)
# print(data.shape, data.dtype)
# print(data[:1000])  # print the first 1000 characters as integers

# split training and validation data
n = int(0.9 * len(data))
train_data = data[:n] # 90% for training
val_data = data[n:] # 10% for validation

# define block size to identify token and its target token for training the model
train_data[:block_size+1]


x= train_data[:block_size]
y= train_data[1:block_size+1]   
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
   # print(f"when input is {context} the target: {target}")

# data loading function to generate batches of data for training and validation
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # select random starting indices for the batch
    x = torch.stack([data[i:i+block_size] for i in ix]) # 
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(torch.device), y.to(torch.device)
    return x, y

# xb, yb = get_batch('train')
# print("inputs:")
# print(xb)
# print(xb.shape)
# print("targets:")
# print(yb)
# print(yb.shape)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Create a nural network model using bigram language model architecture


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B = batch_size, T = time_steps, C = channel_size)
        
        if targets is None:
            loss = None
        else:
            # reshape logits and targets to calculate loss
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(torch.device)


# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=torch.device)
print(decoder(m.generate(idx=context, max_new_tokens=500)[0].tolist()))



