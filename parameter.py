from dataclasses import dataclass


@dataclass
class param():
    model_name = 't5-base'
    result_save_path = './model_save/prompt_model_pretrain.pt'
    batch_size = 16
    max_len = 100
    split_size = 1
    val_split_size = 0.2
    def parameter_dict(self):
        parameter = {'parameter': {'model_name':self.model_name, 'batch_size':self.batch_size, 'max_len':self.max_len, 'split_size':self.split_size}}
        return parameter

    
if __name__ == '__main__':
    test_parm = param()
    print(test_parm.parameter_dict())