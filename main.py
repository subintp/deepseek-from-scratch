import torch
from attention import SelfAttention

def main():
    torch.manual_seed(789)

    # Sequence of 6 tokens, each with 3 dimensions
    inputs = torch.tensor([
      [0.43, 0.15, 0.89],  # Your
      [0.55, 0.87, 0.66],  # journey
      [0.57, 0.85, 0.64],  # starts
      [0.22, 0.58, 0.33],  # with
      [0.77, 0.25, 0.10],  # one
      [0.05, 0.80, 0.55],  # step
    ])

    # Add batch dimension → shape becomes (1, 6, 3)
    inputs = inputs.unsqueeze(0)
    print("Input shape:", inputs.shape)

    d_in = 3
    d_out = 2

    # Instantiate self-attention module
    sa = SelfAttention(d_in, d_out, dropout=0.0)   # disable dropout for testing

    # Run the forward pass
    output = sa(inputs)

    print("\nOutput shape:", output.shape)
    print("\nOutput tensor:\n", output)


if __name__ == "__main__":
    main()
