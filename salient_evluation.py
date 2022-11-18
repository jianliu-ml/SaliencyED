import torch
import math
import numpy
import pickle

from typing import List

from torch import backends
from allennlp.nn import util
from allennlp.common.util import JsonDict, sanitize
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import config_maven as config

from dataset import Dataset
from model import SentenceClassification, get_type_rep
from utils import save_model, load_model


def load_dataset():
    filename = 'data/data_maven.pk'
    data = pickle.load(open(filename, 'rb'))
    return data['train'], data['val'], data['test']


class SaliencyInterpreter(object):
    def __init__(self, model):
        self.model = model
    
    def _compute_loss(self, instance):
        data_x, data_x_mask, data_labels = instance
        loss = self.model(data_x, data_x_mask, data_labels)
        return loss

    def saliency_interpret(self, inputs):
        raise NotImplementedError("Implement this for saliency interpretations")

    @staticmethod
    def _aggregate_token_embeddings(
        embeddings_list: List[torch.Tensor], token_offsets: List[torch.Tensor]
    ) -> List[numpy.ndarray]:
        if len(token_offsets) == 0:
            return [embeddings.cpu().numpy() for embeddings in embeddings_list]
        aggregated_embeddings = []
        for embeddings, offsets in zip(embeddings_list, token_offsets):
            span_embeddings, span_mask = util.batched_span_select(embeddings.contiguous(), offsets)
            span_mask = span_mask.unsqueeze(-1)
            span_embeddings *= span_mask  # zero out paddings

            span_embeddings_sum = span_embeddings.sum(2)
            span_embeddings_len = span_mask.sum(2)
            # Shape: (batch_size, num_orig_tokens, embedding_size)
            embeddings = span_embeddings_sum / torch.clamp_min(span_embeddings_len, 1)

            # All the places where the span length is zero, write in zeros.
            embeddings[(span_embeddings_len == 0).expand(embeddings.shape)] = 0
            aggregated_embeddings.append(embeddings.numpy())
        return aggregated_embeddings


class IntegratedGradient(SaliencyInterpreter):

    def saliency_interpret(self, inputs):
        # Convert inputs to labeled instances

        labeled_instances = inputs   #####################  <---- should be the predicted results

        instances_with_grads = dict()
        for idx, instance in enumerate(labeled_instances):
            # Run integrated gradients
            grads = self._integrate_gradients(instance)

            # Normalize results
            for key, grad in grads.items():
                # The [0] here is undo-ing the batching that happens in get_gradients.
                embedding_grad = numpy.sum(grad[0], axis=1)
                norm = numpy.linalg.norm(embedding_grad, ord=1)
                normalized_grad = [math.fabs(e) / norm for e in embedding_grad]
                grads[key] = normalized_grad

            instances_with_grads["instance_" + str(idx + 1)] = grads

        return sanitize(instances_with_grads)

    def _register_hooks(self, alpha: int, embeddings_list: List, token_offsets: List):

        def forward_hook(module, inputs, output):
            # Save the input for later use. Only do so on first call.
            if alpha == 0:
                embeddings_list.append(output.squeeze(0).clone().detach())

            output.mul_(alpha)

        # Register the hooks
        handles = []
        embedding_layer = util.find_embedding_layer(self.model)
        handles.append(embedding_layer.register_forward_hook(forward_hook))
        return handles


    def _integrate_gradients(self, instance):
        """
        Returns integrated gradients for the given [`Instance`](../../data/instance.md)
        """
        ig_grads: Dict[str, Any] = {}

        # List of Embedding inputs
        embeddings_list: List[torch.Tensor] = []
        token_offsets: List[torch.Tensor] = []

        # Use 10 terms in the summation approximation of the integral in integrated grad
        steps = 10

        # Exclude the endpoint because we do a left point integral approximation
        for alpha in numpy.linspace(0, 1.0, num=steps, endpoint=False):
            handles = []
            # Hook for modifying embedding value
            handles = self._register_hooks(alpha, embeddings_list, token_offsets)

            try:
                grads = self.get_gradients(instance)[0]
            finally:
                for handle in handles:
                    handle.remove()

            # Running sum of gradients
            if ig_grads == {}:
                ig_grads = grads
            else:
                for key in grads.keys():
                    ig_grads[key] += grads[key]

        # Average of each gradient term
        for key in ig_grads.keys():
            ig_grads[key] /= steps

        # Gradients come back in the reverse order that they were sent into the network
        embeddings_list.reverse()
        token_offsets.reverse()
        embeddings_list = self._aggregate_token_embeddings(embeddings_list, token_offsets)

        # Element-wise multiply average gradient by the input
        for idx, input_embedding in enumerate(embeddings_list):
            key = "grad_input_" + str(idx + 1)
            ig_grads[key] *= input_embedding

        return ig_grads

    
    def get_gradients(self, instance):

        # set requires_grad to true for all parameters, but save original values to
        # restore them later
        original_param_name_to_requires_grad_dict = {}
        for param_name, param in self.model.named_parameters():
            original_param_name_to_requires_grad_dict[param_name] = param.requires_grad
            param.requires_grad = True

        embedding_gradients: List[Tensor] = []
        hooks: List[RemovableHandle] = self._register_embedding_gradient_hooks(embedding_gradients)

        with backends.cudnn.flags(enabled=False):
            loss = self._compute_loss(instance)
            for p in self.model.parameters():
                p.grad = None
            loss.backward()

        for hook in hooks:
            hook.remove()

        grad_dict = dict()
        for idx, grad in enumerate(embedding_gradients):
            key = "grad_input_" + str(idx + 1)
            grad_dict[key] = grad.detach().cpu().numpy()

        # restore the original requires_grad values of the parameters
        for param_name, param in self.model.named_parameters():
            param.requires_grad = original_param_name_to_requires_grad_dict[param_name]

        return grad_dict, []


    def _register_embedding_gradient_hooks(self, embedding_gradients):

        def hook_layers(module, grad_in, grad_out):
            grads = grad_out[0]
            embedding_gradients.append(grads)

        hooks = []
        embedding_layer = util.find_embedding_layer(self.model)
        hooks.append(embedding_layer.register_backward_hook(hook_layers))
        return hooks


if __name__ == '__main__':
    device = 'cuda:0'

    batch_size = 80
    train_data, val_data, test_data = load_dataset()
    print(len(train_data))

    result = []
    for elem in test_data:
        # print(elem)
        a, b, c, d, e, f, g, h = elem
        if sum(c) == 0:
            result.append(elem)
        else:
            for x in d:
                t = [0] * len(c)
                t[x] = 1
                y = {x: d[x]}
                result.append([a, b, t, y, e, f, g, h])
    # train_data = result
    # print(len(train_data))

    my_dataset = Dataset(batch_size, 150, result)
    # val_dataset = Dataset(batch_size, 100, val_data)
    # test_dataset = Dataset(batch_size, 120, test_data)

    rep = get_type_rep().detach_().to(device)
    model = SentenceClassification(config.bert_dir, len(config.idx2tag), rep)
    load_model(model, 'model/maven_sentence9.ckp')
    model.to(device)

    ig = IntegratedGradient(model)

    salience_data = []

    for batch in my_dataset.get_tqdm(device, False):
        data_x, data_x_mask, data_labels, _, _, appendix = batch

        instances = [[data_x[i], data_x_mask[i], data_labels[i]] for i in range(len(data_x))]
        instances = [list(map(lambda x: x.unsqueeze(0), x)) for x in instances]
        # print(instances[0])

        res = ig.saliency_interpret(instances)
        salience_data.append([res, data_x, appendix])
        # for e, d, p in zip(res, data_x, appendix):

        #     x = p[0]
        #     if x:
        #         print(p)
        #         words = config.tokenizer.convert_ids_to_tokens(d)
        #         words = list(filter(lambda x: x!='[PAD]', words))

        #         probs = res[e]['grad_input_1']
        #         probs = numpy.around(probs, 3)

        #         for idx, (w, p) in enumerate(zip(words, probs)):
        #             print(idx, w, p)
        #         print('---')

    f = open('maven_salience_ind.pk','wb')
    pickle.dump(salience_data, f)
    f.close()
