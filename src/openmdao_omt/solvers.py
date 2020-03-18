import openmdao.api as om


class DiscreteBroydenSolver(om.BroydenSolver):
    def _disallow_discrete_outputs(self):
        pass
