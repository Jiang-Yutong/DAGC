from MFL.model.CNN import CNN1, CNN3
def get_model(model_name):
    if model_name == 'CNN1':
        return CNN1()
    elif model_name == 'CNN3':
        return CNN3()