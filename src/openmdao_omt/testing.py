import numpy as np
import scipy as sp
import openmdao.api as om
import random

from . import VariableType


def hyperplane_coefficients(points):
    A = np.c_[points[:, :-1], np.ones(points.shape[0])]
    B = points[:, -1]
    coeff, _, _, _ = sp.linalg.lstsq(A, B)
    return coeff


def is_assertion_error(err, *args):
    return issubclass(err[0], AssertionError)


class NoiseComponent(om.ExplicitComponent):
    def setup(self):
        self.add_output("y")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["y"] = random.random()


class PassthroughComponent(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("shape")
        self.options.declare("in_type", types=VariableType)
        self.options.declare("out_type", types=VariableType)

    def setup(self):
        if self.options["in_type"] is VariableType.CONTINUOUS:
            self.add_input("in", shape=self.options["shape"])
        else:
            self.add_discrete_input("in", None)

        if self.options["out_type"] is VariableType.CONTINUOUS:
            self.add_output("out", shape=self.options["shape"])
        else:
            self.add_discrete_output("out", None)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        inputs_ = inputs if self.options["in_type"] is VariableType.CONTINUOUS else discrete_inputs
        outputs_ = outputs if self.options["out_type"] is VariableType.CONTINUOUS else discrete_outputs

        outputs_["out"] = inputs_["in"]
