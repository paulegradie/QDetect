
class ArgContainer(object):
    def __init__(self, gen_pickle=None, static_pickle=None, model_dir=None):
        self.gen_pickle = gen_pickle
        self.static_pickle = static_pickle
        self.model_dir = model_dir

        assert self.gen_pickle or self.static_pickle, "must provide data pickle"
        assert model_dir, 'must provide model dir'
        assert 'train' not in self.command, 'Options: [eval | predict]'