from FSCIL.tree_model.gp_model import GP_Model
from FSCIL.tree_model.gp_model_gibbs import GP_Model_Gibbs
from torch import nn
from utils import *


class Node(nn.Module):
    def __init__(self):
        super(Node, self).__init__()
        self.left_child = None
        self.right_child = None
        self.model = None
        self.device = None
        self.id = 0
        self.depth = 0
        self.num_data = 1
        self.init_node()

    def init_node(self):
        self.X_support = None
        self.Y_support = None
        self.state = None
        self.old_to_new = {}
        self.new_to_old = {}
        self.classes = []

    def set_child(self, node, child=0):
        if child == 0:  # left child
            self.left_child = node
        elif child == 1:  # right child
            self.right_child = node
        else:
            raise NotImplementedError("not a valid child")

    def map_old_to_new_lbls(self, Y):
        new_to_old = {}
        new_lbls = torch.clone(Y)
        for old_class, new_class in self.old_to_new.items():
            if new_class in new_to_old:  # if key exists append to the list of corresponding old classes
                new_to_old[new_class] = new_to_old[new_class] + [old_class]
            else:
                new_to_old[new_class] = [old_class]
            # assign new label. add constant to prevent from running over new class in following iterations
            new_lbls[new_lbls == old_class] = new_class + 1000

        new_lbls -= 1000
        return new_lbls, new_to_old


class Node_VI(Node):

    def set_data(self, Y_support, old_to_new, node_orig_to_new_idx):

        self.classes, _ = torch.sort(torch.unique(Y_support))
        self.old_to_new = old_to_new

        Y_supp_new_lbls, new_to_old = self.map_old_to_new_lbls(Y_support)
        self.new_to_old = new_to_old
        self.node_orig_to_new_idx = node_orig_to_new_idx

    def set_model(self, kernel_function, num_inducing_points, natural_lr,
                  outputscale, lengthscale, dtype, num_data):

        num_classes = 2
        self.num_data = num_data

        # number of inducing points is at most the number of data points
        self.model = GP_Model(kernel_func=kernel_function, num_classes=num_classes, dtype=dtype,
                           num_inducing_points=num_inducing_points, num_data=num_data,
                           natural_lr=natural_lr)

        self.model.model._set_params(outputscale=outputscale,
                                     lengthscale=lengthscale)

        self.model.to(self.device)

    def train_loop(self, X, Y, batch_idx, Z, to_print):

        train_data = torch.cat((Z.reshape(-1, Z.shape[-1]), X), dim=0)
        loss = - self.model.forward_mll(train_data, Y, batch_idx, to_print=to_print)

        avg_loss = loss.item() / self.num_data
        if to_print:
            logging.info(f"Loss: {loss.item():.5f}, Avg. Loss: {avg_loss}")

        self.model.ELBO.update()  # update natural parameters
        return loss / self.num_data


class Node_Gibbs(Node):

    def set_data(self, Y_support, old_to_new):

        self.classes, _ = torch.sort(torch.unique(Y_support))
        self.old_to_new = old_to_new

        Y_supp_new_lbls, new_to_old = self.map_old_to_new_lbls(Y_support)
        self.new_to_old = new_to_old

    def set_model(self, kernel_function, num_steps, num_draws,
                  outputscale, lengthscale, num_data):

        num_classes = 2
        self.num_data = num_data

        # number of inducing points is at most the number of data points
        self.model = GP_Model_Gibbs(kernel_func=kernel_function, num_classes=num_classes,
                                 num_data=num_data, num_steps=num_steps, num_draws=num_draws)

        self.model.model._set_params(outputscale=outputscale,
                                     lengthscale=lengthscale)

        self.model.to(self.device)

    def train_loop(self, X, Y, to_print):
        loss = self.model.forward_mll(X, Y, to_print=to_print)
        return loss