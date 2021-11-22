

from modeling_transformer import Transformer


model = Transformer(
        vocab_size=25000,
        model_dim=512,
        hidden_dim=2048,
        nheads=8,
        max_len=512,
        depth=6,
    )

src_batch = []# Tensor with shape (batch_size, src_sentence_length)
tgt_batch = []# Tensor with shape (batch_size, tgt_sentence_length)

outputs = model(src_batch, tgt_batch)
print('*****:', outputs)