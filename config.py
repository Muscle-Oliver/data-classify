import torch
import nni

class DefaultConfig:
    training=False
    model_name = 'RNN' #RNN_FIXED 效果略差
    
    train_path = './DC/data/train.txt'
    dev_path = './DC/data/test.txt'
    test_path = './DC/data/test.txt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_path = './DC/saved_dict/'+ model_name + '_best.pth'
    require_improvement = 100000000

    # use_model_blending = False
    # weight_1 = 0.5
    # weight_2 = 0.5

    # other_index_1 = 2
    # other_index_2 = 3

    load_model_path = None

    batch_size = 128  # batch size
    num_workers = 0


    # source_data_folder = 'source1'

    background_data_path = 'background_shuffle_100000.txt'
    # dataset_selfdefine = None
    # dataset_selfdefine = 'self-defined.txt'
    # save_folder = 'demo2_out'

    labels2index_path = './DC/data/labels2index.pkl'
    vocab_path = './DC/data/vocab.pkl'

    max_epoch = 20
    max_iter = 500000
    lr = 0.01  # initial learning rate
    # lr = 5e-4
    #weight_decay = 1e-4
    weight_decay = 5e-5 #建议参数
    embed_size = 300
    drop_prop = 0.5
    classes = 11
    # max_len = 50

    use_lrdecay = True
    lr_decay = 0.95
    n_epoch = 2

    use_clip = True
    grad_clip = 2

    recurrent_hidden_size = 128
    num_layers = 2
    bidirectional = True

    # use_pretrained_word_vector = True
    word_vector_path = 'data/sgns.sogou.char'
    frozen = True
    pretrained_embedding_matrix_path = './DC/data/embedding_SougouNews_char.npz'
    # pretrained_embedding_matrix_path = None

    #text = 'hello'
    text = None
    prediction_path = "data/prediction.txt"
    #prediction_path = None

    predict_pad = False
    predict_batchsize = 2500
    predict_result_path = 'predict_result.txt'
    
    try:
        # 超参搜索 NNI HPO by hjy
        '''
        nni params as:
        search_space = {
                "lr": {"_type": "loguniform", "_value": [0.0001, 0.01]},
                "weight_decay": {"_type": "loguniform", "_value": [0.000005, 0.00005, 0.0005]},
                "max_epoch": {"_type": "choice", "_value": [5, 10, 15, 20]},
                "batch_size": {"_type": "choice", "_value": [64, 128, 256]},
            }
        '''
        params = nni.get_next_parameter()
        #params = {"lr": 1e-4, "weight_decay": 5e-5, "max_epoch": 5, "batch_size": 128}
        lr = params["lr"]
        weight_decay = params["weight_decay"]
        max_epoch = params["max_epoch"]
        batch_size = params["batch_size"]
        # 超参搜索 NNI HPO by hjy
    except:
        pass

def parse(self, kwargs):
    '''
    '''
    for k, v in kwargs.items():
        if not hasattr(self, k):
            warnings.warn("Warning: opt has not attribut %s" % k)
        setattr(self, k, v)

    # print('user config:')
    # for k, v in self.__class__.__dict__.items():
    #     if not k.startswith('__'):
    #         print(k, getattr(self, k))

DefaultConfig.parse = parse
opt = DefaultConfig()
