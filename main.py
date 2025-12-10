import torch
from attention import SelfAttention, MultiHeadedAttention

def main():
    torch.manual_seed(123)

    # Define the tensor with 3 rows and 6 columns
    inputs = torch.tensor([
        [0.43, 0.15, 0.89, 0.55, 0.87, 0.66], # Row 1
        [0.57, 0.85, 0.64, 0.22, 0.58, 0.33], # Row 2
        [0.77, 0.25, 0.10, 0.05, 0.80, 0.55]  # Row 3
    ])

    batch = torch.stack((inputs, inputs), dim=0)
    print("Batch shape:", batch.shape)

    batch_size, num_tokens, d_in = batch.shape
    d_out = 6
    num_heads = 2

    # Instantiate multi-headed attention module
    mha = MultiHeadedAttention(d_in, d_out, dropout=0.0, num_heads=num_heads)

    # Run the forward pass
    context_vecs = mha(batch)

    # Print the output
    print("\nOutput shape:", context_vecs.shape)
    print("\nOutput tensor:\n", context_vecs)


if __name__ == "__main__":
    main()
