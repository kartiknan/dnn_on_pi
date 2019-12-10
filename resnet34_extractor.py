import fastai
import torch
from torchvision.models import resnet34
from torch import nn
from fastai.vision.learner import create_head, num_features_model


def get_model_classes(classes_path):
    labels = ''
    with open(classes_path, "r") as rfh:
        labels = rfh.read().splitlines()
    return labels


def create_model(num_classes=None):
    """Create a fastai compatible torch model.

    This model is a sequential model with body and head children.

    If num_classes is not provided, it will use the torch models
    default head (ie. for 1k imagenet classes)
    """

    model = new_model()

    if num_classes:
        body = list(model.children())[0]
        nf = num_features_model(body) * 2
        head = create_head(nf, num_classes, None, ps=0.5, bn_final=False)
        model = nn.Sequential(body, head)

    return model


def new_model():
    """Create a fastai compatible pretrained resnet model.

    The head of this model will be for 1k imagenet class classification
    """
    model = resnet34(pretrained=True)
    children = list(model.children())
    body = nn.Sequential(*children[:-2])
    head = nn.Sequential(*children[-2:])
    return nn.Sequential(body, head)


class ResNet34Extractor(torch.nn.Module):
    """ ResNet model that outputs relevant layers for PCA

    Per layer methods exist to allow for staged computation of forward
    passes, so any per layer processing (eg. PCA) can be performed in
    a more memory efficient manner.

    Note: A full forward pass is equivalent to:
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.fc(x)
        x = self.prediction(x)
    """

    def __init__(self, state_dict=None, classes=None):
        super(ResNet34Extractor, self).__init__()

        self.classes = classes
        if self.classes:
            num_classes = len(self.classes)
            self.base_model = create_model(num_classes=num_classes)
            self.pretrained = False
        else:
            self.base_model = create_model()
            self.pretrained = True

        if state_dict:
            n_layers = len(state_dict.keys())
            n_layers_ok = 0
            for key, value in state_dict.items():
                try:
                    self.base_model.load_state_dict({key: value}, strict=False)
                    n_layers_ok += 1
                except:
                    print(f'Failed to load weights for layer {key}')
            print(
                f'Loaded {n_layers_ok} / {n_layers} layers from state dict')

            last_layer = list(state_dict.keys())[-1]
            n_state_classes = state_dict[last_layer].size()[0]
            if not self.classes or (n_state_classes != len(self.classes)):
                self.pretrained = True

        # We use .eval() so that dropout isn't applied
        self.base_model.eval()

        if torch.cuda.is_available():
            self.base_model = self.base_model.cuda()

        self.base_model_body, self.base_model_head = \
            list(self.base_model.children())

        self.body_children = [child for _, child in
                              self.base_model_body.named_children()]
        self.head_children = [child for _, child in
                              self.base_model_head.named_children()]

    def get_head_model(self):
        """Return a Sequential model for the head"""
        return torch.nn.Sequential(
            *self.head_children[:-1],
            Reshape(),
            self.head_children[-1]
        )

    def get_fc_model(self):
        """Return a Sequential model for the head"""
        return torch.nn.Sequential(
            Reshape(),
            self.head_children[-1]
        )

    def l1(self, x):
        """
        :param x: input images as tensors
        :return: hidden layer activations for l1
        """
        for layer in self.body_children[:5]:
            x = layer(x)
        return x

    def l2(self, x):
        """
        :param x:  hidden layer activations for l1
        :return: hidden layer activations for l2
        """
        return self.body_children[5](x)

    def l3(self, x):
        """
        :param x:  hidden layer activations for l2
        :return: hidden layer activations for l3
        """
        return self.body_children[6](x)

    def l4(self, x):
        """
        :param x:  hidden layer activations for l3
        :return: hidden layer activations for l4
        """
        return self.body_children[7](x)

    def fc(self, x):
        """
        :param x:  hidden layer activations for l4
        :return: hidden layer activations for fc
        """
        for layer in self.head_children[:-1]:
            x = layer(x)
        # x = x.view(x.size(0), -1)
        return x

    def predictions(self, x):
        """
        :param x:  hidden layer activations for fc
        :return: final layer outputs
        """
        x = self.head_children[-1](x)
        return x

    def batch_predict(self, batch):
        with torch.no_grad():
            tsm = torch.nn.Softmax(dim=1)
            batch = batch.cuda() if torch.cuda.is_available() else batch
            f_body = self.base_model_body.forward(batch)
            f_fc = torch.squeeze(self.fc(f_body))
            out = tsm(self.predictions(f_fc))
            return out
