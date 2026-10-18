"""A numerically stable independent Bernoulli gate conditioned to be nonempty."""

import torch
from torch.distributions import Bernoulli, Categorical
from torch.nn import functional as F


class NonemptyBernoulli:
    """Distribution over bit vectors with at least one selected head.

    The first selected index has probability proportional to the probability
    that all earlier bits are zero and that bit is one. Later bits remain
    independent Bernoulli variables. This samples exactly without rejection,
    including when every logit is very negative.
    """

    def __init__(self, logits: torch.Tensor):
        if logits.ndim != 1 or logits.numel() == 0:
            raise ValueError("gate logits must be a nonempty 1-D tensor")
        if not torch.isfinite(logits).all():
            raise ValueError("gate logits must be finite")
        self.logits = logits
        self._log_p = F.logsigmoid(logits)
        self._log_q = F.logsigmoid(-logits)
        earlier_zero = torch.cat((logits.new_zeros(1), self._log_q.cumsum(0)[:-1]))
        self._first_logits = earlier_zero + self._log_p
        # Sum over mutually exclusive events: first selected head is i.
        self.log_normalizer = torch.logsumexp(self._first_logits, dim=0)

    def sample(self) -> torch.Tensor:
        first = int(Categorical(logits=self._first_logits).sample().item())
        gate = torch.zeros_like(self.logits, dtype=torch.bool)
        gate[first] = True
        if first + 1 < gate.numel():
            gate[first + 1 :] = Bernoulli(logits=self.logits[first + 1 :]).sample().bool()
        return gate

    def mode(self) -> torch.Tensor:
        """Most likely nonempty gate, with deterministic lowest-index ties."""
        gate = self.logits > 0
        if not bool(gate.any()):
            gate = gate.clone()
            gate[int(self.logits.argmax())] = True
        return gate

    def log_prob(self, gate: torch.Tensor) -> torch.Tensor:
        if gate.shape != self.logits.shape or gate.dtype != torch.bool:
            raise ValueError("gate must be a Boolean vector matching logits")
        if not bool(gate.any()):
            raise ValueError("all-zero gate is excluded")
        return torch.where(gate, self._log_p, self._log_q).sum() - self.log_normalizer

    def entropy(self) -> torch.Tensor:
        """Exact entropy of the conditioned gate, computed without enumeration."""
        # First selected index is categorical; bits after it are independent.
        first_probability = (self._first_logits - self.log_normalizer).exp()
        first_entropy = -(first_probability * (self._first_logits - self.log_normalizer)).sum()
        later_probability = torch.cat((self.logits.new_zeros(1), first_probability.cumsum(0)[:-1]))
        bernoulli_entropy = -(self._log_p.exp() * self._log_p + self._log_q.exp() * self._log_q)
        return first_entropy + (later_probability * bernoulli_entropy).sum()


class FeasibleBernoulli:
    """Independent Bernoulli logits conditioned on a finite feasible gate set."""

    def __init__(self, logits: torch.Tensor, support: torch.Tensor):
        if logits.ndim != 1 or not bool(torch.isfinite(logits).all()):
            raise ValueError("gate logits must be a finite 1-D tensor")
        if (support.ndim != 2 or support.shape[1] != logits.numel() or
                support.shape[0] == 0 or support.dtype != torch.bool or
                support.device != logits.device or not bool(support.any(dim=1).all())):
            raise ValueError("support must contain nonempty Boolean gates matching logits")
        if torch.unique(support, dim=0).shape[0] != support.shape[0]:
            raise ValueError("support contains duplicate gates")
        self.logits = logits
        self.support = support
        log_p, log_q = F.logsigmoid(logits), F.logsigmoid(-logits)
        self._unnormalized = torch.where(support, log_p, log_q).sum(dim=1)
        self.log_normalizer = torch.logsumexp(self._unnormalized, dim=0)

    def sample(self) -> torch.Tensor:
        index = Categorical(logits=self._unnormalized).sample()
        return self.support[index]

    def mode(self) -> torch.Tensor:
        return self.support[self._unnormalized.argmax()]

    def log_prob(self, gate: torch.Tensor) -> torch.Tensor:
        if gate.shape != self.logits.shape or gate.dtype != torch.bool or gate.device != self.logits.device:
            raise ValueError("gate must be a Boolean vector matching logits")
        if not bool(gate.any()):
            raise ValueError("all-zero gate is excluded")
        match = (self.support == gate).all(dim=1)
        if not bool(match.any()):
            raise ValueError("gate has no feasible mapping")
        return self._unnormalized[torch.nonzero(match, as_tuple=False)[0, 0]] - self.log_normalizer

    def entropy(self) -> torch.Tensor:
        log_probability = self._unnormalized - self.log_normalizer
        probability = log_probability.exp()
        return -(probability * log_probability).sum()
