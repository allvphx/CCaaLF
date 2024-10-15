from typing import Optional

import torch
from botorch.test_functions.base import BaseTestProblem
from policy import *


class CCControl(BaseTestProblem):
    def __init__(
            self,
            command,
            seed,
            log_dir,
            writer,
            noise_std: Optional[float] = None,
            negate: bool = False
    ) -> None:
        self.dim = 4 * MAX_STATE
        self._bounds = [[0, cap[i] - eps] for i in range(len(cap))]
        print("the transpose = ", self._bounds)
        self.num_objectives = 1
        self.command = command
        self.seed = seed
        self.fin_log_dir = log_dir
        self.writer = writer

        super().__init__(
            noise_std=noise_std,
            negate=negate,
        )

        self.categorical_dims = np.arange(self.dim)

    def evaluate_true(self, X):
        res = torch.stack([self._compute(x) for x in X]).to(X)
        return res

    def _compute(self, x):
        evaluation = black_box_function(command=self.command, writer=self.writer, fin_log_dir=self.fin_log_dir, x=x)
        return torch.tensor(evaluation)
