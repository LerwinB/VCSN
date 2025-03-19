import torch
import torch.nn as nn

class ComplexContrastiveLoss(nn.Module):
    def __init__(self, delta=1e-8):
        super(ComplexContrastiveLoss, self).__init__()
        self.delta = delta

    def forward(self, h_t, g_a):
        B = h_t.size(0)
        
        consM = torch.exp(h_t@g_a.T)
        positive_term =torch.diag(consM)
        neg_term1 = torch.sum(consM,dim=0)
        neg_term2 = torch.sum(consM,dim=1)
        
        # Calculate the log terms
        log_term_1 = torch.log(positive_term / (positive_term + neg_term1))
        log_term_2 = torch.log(positive_term / (positive_term + neg_term2))
        
        # Calculate the final loss
        loss = -torch.mean(log_term_1 + log_term_2)
        
        return loss

# Example usage:
# h_t, g_a, h_imp, and g_imp should be the output from your model's different branches
h_t = torch.randn(32, 128)  # Example batch of feature vectors
g_a = torch.randn(32, 128)  # Example batch of auxiliary feature vectors
h_imp = torch.randn(32, 32, 128)  # Example batch of imposter feature vectors for h
g_imp = torch.randn(32, 32, 128)  # Example batch of imposter feature vectors for g

criterion = ComplexContrastiveLoss(delta=1e-8)
loss = criterion(h_t, g_a, h_imp, g_imp)
print(loss.item())
