import autokoopman.core.pipeline as apipe
import autokoopman.core.hyperparameter as ahyp


def test_compositionality():
    """test that the compositionality works"""

    class Succ(apipe.Pipeline):
        def execute(self, x, _):
            return x + 1

    class Source(apipe.Pipeline):
        def execute(self, x, __):
            return x

    class Identity(apipe.Pipeline):
        def execute(self, x, _):
            return x

    head = Source("head")
    fork1 = Succ("fork1")

    fork1 |= Identity("stage3")
    fork1 |= Succ("stage4")

    head |= Succ("stage2")
    head |= fork1

    assert head.run(0, []) == (1, (1, 2))
    assert head.run(5, []) == (6, (6, 7))
