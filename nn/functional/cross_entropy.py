from engine.tensor import Tensor, arange, log, exp

def cross_entropy(logits: Tensor, true_labels: Tensor) -> Tensor:
    '''Like torch.nn.functional.cross_entropy with reduction='none'.

    logits: shape (batch, classes)
    true_labels: shape (batch,). Each element is the index of the correct label in the logits.

    Return: shape (batch, ) containing the per-example loss.
    '''
    n_batch, n_class = logits.shape
    true = logits[arange(0, n_batch), true_labels]
    return -log(exp(true) / exp(logits).sum(1))