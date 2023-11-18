from physics import Entropy

class EntropicWrapper(nn.Module):
    def __init__(self):
        """Entropic Wrapper
        """

        self.entropy = Entropy()

    def forward(self, *args, **kwargs):
        """Adjust forward pass to include gradient ascend on the entropy

        Args:
            data (_type_): _description_
        """

        out = super().forward(*args, **kwargs)

        out += self.entropy.gradient_entropy()

        return out

    # TODO: update the embedding after forward pass I guess? So here we should see how we actually use the wrapper