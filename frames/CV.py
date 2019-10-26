from . import DeeplearningClassification2

'''
网格搜索
'''

class AnnCV():
    def __init__(self, model, parameters, X_train, y_train, X_test, y_test):
        self.model = model
        self.parameters = parameters

        self.X_train, self.y_train, self.X_test, self.y_test = X_train, y_train, X_test, y_test

        self.is_standard = False
        self.Dimensionality_reduction_method = None
        self.save_path = './Classification_results/ANN'
        self.device = 0
        self.use_more_gpu = False

    def fit(self):


        for key, value in self.parameters.items():
            if key == 'hidden_layers':
                hls = value
            if key == 'learning_rate':
                lrs = value
            if key == 'dropout':
                ds = value
            if key == 'activate_function':
                af = value
            if key == 'weight_decay':
                wd = value
            if key == 'epoch':
                es = value
            if key == 'batch_size':
                bs = value
            if key == 'is_standard':
                self.is_standard = value
            if key == 'Dimensionality_reduction_method':
                self.Dimensionality_reduction_method = value
            if key == 'save_path':
                self.save_path = value
            if key == 'device':
                self.device = value
            if key == 'use_more_gpu':
                self.use_more_gpu = value

        for h in hls:
            for lr in lrs:
                for d in ds:
                    for a in af:
                        for w in wd:
                            for e in es:
                                for b in bs:
                                    ann = self.model(hidden_layers=h, learning_rate=lr, dropout=d,
                                                     activate_function=a, weight_decay=w, device=self.device,
                                                     use_more_gpu=self.use_more_gpu, epoch=e, batch_size=b,
                                                     is_standard=self.is_standard,
                                                     Dimensionality_reduction_method=self.Dimensionality_reduction_method,
                                                     save_path=self.save_path)
                                    print('Now...hidden layers: {}, learning rate: {}, dropout: {}, '
                                          'activate function: {}, weight decay: {}, epoch: {}, batch size: {}'.format(
                                        h, lr, d, a, w, e, b
                                    ))

                                    ann.fit(self.X_train, self.y_train, self.X_test, self.y_test)
                                    ann.score()
                                    ann.save()