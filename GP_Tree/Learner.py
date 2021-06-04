import torch.nn as nn
from backbone import ResNet18
from GP_Tree.tree import BinaryTree
from utils import *
from sklearn.cluster import KMeans


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.tree = None
        self.NN_classifier = True
        self.criterion = nn.CrossEntropyLoss()

    def _init_Xbar(self, X, Y):
        raise NotImplementedError("not yet implemented")

    def forward(self, x, y, x_idx, to_print=True):
        raise NotImplementedError("not yet implemented")

    def forward_subtree(self, *args):
        raise NotImplementedError("not yet implemented")

    def forward_eval(self, x, y, num_classes):
        raise NotImplementedError("not yet implemented")

    def get_features(self, x):
        return self.features(x, classify=False)

    def build_base_tree(self, X, Y, X_idx):
        raise NotImplementedError("not yet implemented")

    def expend_tree(self, *args):
        raise NotImplementedError("not yet implemented")

    def disable_bn(self):
        for m in self.features.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.eval()


class ModelBinaryTree(Model):
    def __init__(self, args, device, pretrained=True):
        super(ModelBinaryTree, self).__init__(args)
        self.features = ResNet18(dims=args.NN_layers + [args.N_way[0]], args=args, pretrained=pretrained)
        Xbar_dim = (sum(self.args.N_way), self.args.num_inducing_points, self.args.NN_layers[-1])
        if self.args.learn_location:
            self.Xbar = nn.Parameter(torch.randn(Xbar_dim), requires_grad=True)
        else:
            self.Xbar = torch.randn(Xbar_dim).to(device)

    def _init_Xbar(self, X, Y, kmeans=True):
        with torch.no_grad():
            classes = torch.unique(Y)
            num_inducing = self.args.num_inducing_points
            for c in sorted(detach_to_numpy(classes).tolist()):
                X_c, Y_c = pytorch_take(X, Y, [c])

                # Inducing points locations
                if num_inducing < X_c.shape[0] and kmeans:
                    kmeans = KMeans(n_clusters=num_inducing, n_init=5, max_iter=200, random_state=42). \
                        fit(detach_to_numpy(X_c))
                    Xbar = torch.tensor(kmeans.cluster_centers_, dtype=X_c.dtype).to(X.device)
                else:
                    # expend X arbitrarily by 4 and take the first #inducing locations
                    Xbar = X_c.clone().repeat(4, 1)[:num_inducing, ...]

                self.Xbar[c, ...].copy_(Xbar)

    def forward(self, x, y, x_idx, to_print=True):

        z = self.features(x, classify=self.NN_classifier)
        if not self.NN_classifier:
            loss = self.tree.train_tree(z, y, x_idx, self.Xbar, to_print)
        else:
            loss = self.criterion(z, y)

        return loss

    def forward_gibbs(self, z_base, y_base, x_novel_prev_new, y_novel_prev_new, to_print=True):

        z_novel_prev, x_novel = x_novel_prev_new
        y_novel_prev, y_novel = y_novel_prev_new

        z_novel = self.features(x_novel, classify=False)

        if y_novel_prev is not None:
            z = torch.cat((z_base, z_novel_prev, z_novel), dim=0)
            z_y = torch.cat((y_base, y_novel_prev, y_novel), dim=0)
        else:
            z = torch.cat((z_base, z_novel), dim=0)
            z_y = torch.cat((y_base, y_novel), dim=0)

        loss = self.tree.train_right_subtree(z, z_y, to_print)

        return loss

    def forward_eval(self, x, y, num_classes):
        z = self.features(x, classify=self.NN_classifier)
        if not self.NN_classifier:
            preds = self.tree.eval_tree_full_path(z, num_classes, self.Xbar)
            loss = CE_loss(y, preds, num_classes)
        else:
            preds = z
            loss = self.criterion(preds, y)

        return loss, preds

    def gibbs_forward_eval(self, x, y, num_classes):
        z = self.features(x, classify=False)
        preds = self.tree.gibbs_eval_tree_full_path(z, num_classes, self.Xbar)
        loss = CE_loss(y, preds, num_classes)

        return loss, preds

    def build_base_tree(self, X, Y, X_idx):

        # Init Z to datapoints region
        self._init_Xbar(X, Y)

        # Create tree instance
        self.tree = BinaryTree(self.args, X.device)
        # set the root of the tree
        subtree_gp_counter = self.tree.build_tree(self.tree.root, X, Y, X_idx, self.Xbar)
        self.tree.print_tree()
        self.tree.to(X.device)
        self.NN_classifier = False
        return subtree_gp_counter

    def expend_tree(self, X_base, Y_base, X_novel_prev_new, Y_novel_prev_new):

        X_novel_prev, X_novel = X_novel_prev_new
        Y_novel_prev, Y_novel = Y_novel_prev_new

        # get new Xs in feature space
        X_novel_f = self.features(X_novel, classify=False)

        # Init Xbar to datapoints region
        self._init_Xbar(X_novel_f, Y_novel)

        if Y_novel_prev is not None:
            X_novel_f = torch.cat((X_novel_prev, X_novel_f))
            Y_novel = torch.cat((Y_novel_prev, Y_novel))

        # build tree
        subtree_gp_counter = self.tree.expend_right_subtree(Y_base, X_novel_f, Y_novel)
        self.tree.print_tree()

        self.tree.to(Y_base.device)
        return subtree_gp_counter
