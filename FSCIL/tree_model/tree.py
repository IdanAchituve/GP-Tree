from FSCIL.tree_model.node import Node_VI, Node_Gibbs
from FSCIL.tree_model.class_splits import *
from utils import (detach_to_numpy, pytorch_take, pytorch_take2)
import logging
from torch import nn
import torch
from collections import deque
import copy


class BinaryTree(nn.Module):
    def __init__(self, args, device):
        super(BinaryTree, self).__init__()
        self.root = None
        self.args = args
        self.device = device
        self.root = Node_VI()
        self.root.id = 0
        self.root.depth = 0

    def split_func(self, X, Y, *args, **kwargs):
        # method for splitting classes
        return {'Split': Split(Y, 2),
                'MeanSplitKmeans': MeanSplitKmeans(Y, 2, X)
                }

    def get_root(self):
        return self.root

    def build_tree(self, root, X, Y, X_idx, Xbar):
        """
        Build binary tree with GP attached to each node
        """
        # root
        q = deque()

        # push source vertex into the queue
        q.append((root, X, Y, X_idx))
        curr_id = 1
        gp_counter = 0  # for getting avg. loss over the whole tree

        # loop till queue is empty
        while q:
            # pop front node from queue
            root, root_X, root_Y, root_Xidx = q.popleft()
            node_classes, _ = torch.sort(torch.unique(root_Y))
            num_classes = node_classes.size(0)

            # Xbar's of current node
            Xbar_r = Xbar[node_classes, ...]

            # two classes or less - no heuristic for splitting
            split_method = 'MeanSplitKmeans' if num_classes > 2 else 'Split'
            root_old_to_new = \
                self.split_func(detach_to_numpy(root_X),
                                detach_to_numpy(root_Y))[split_method].split()

            node_orig_to_new_idx = {x_i: i for i, x_i in enumerate(sorted(detach_to_numpy(root_Xidx).tolist()))}
            root.set_data(root_Y, root_old_to_new, node_orig_to_new_idx)

            # leaf node
            if num_classes == 1:
                logging.info('Reached a leaf node. Node index: ' + str(root.id) + ' ')
                continue

            # Internal node
            else:
                gp_counter += 1
                num_inducing_inputs = Xbar_r.shape[0] * Xbar_r.shape[1]
                root.set_model(self.args.kernel_function, num_inducing_inputs,
                               self.args.natural_lr, self.args.outputscale, self.args.lengthscale,
                               root_X.dtype, root_X.shape[0])

                left_X, left_Y, left_Xidx = pytorch_take2(root_X, root_Y, root_Xidx, root.new_to_old[0])
                right_X, right_Y, right_Xidx = pytorch_take2(root_X, root_Y, root_Xidx, root.new_to_old[1])
                child_X = [left_X, right_X]
                child_Y = [left_Y, right_Y]
                child_Xidx = [left_Xidx, right_Xidx]

                branches = 2
                for i in range(branches):
                    child = Node_VI()
                    child.id = curr_id
                    curr_id += 1
                    child.depth = root.depth + 1
                    root.set_child(child, i)
                    q.append((child, child_X[i], child_Y[i], child_Xidx[i]))

        return gp_counter

    def expend_right_subtree(self, Y_base, X_novel, Y_novel):
        """
        Expend tree to the right with novel classes
        *_old corresponds to left subtree inducing points
        *_novel corresponds to novel original data
        """

        # build subtree only for novel classes
        root_right = Node_Gibbs()
        gp_counter = self.build_tree_gibbs(root_right, X_novel, Y_novel)

        # loop until finding the root node of base classes
        root_left = self.root
        Y_base_classes, _ = torch.sort(torch.unique(Y_base))
        root_classes, _ = torch.sort(torch.unique(root_left.classes))
        while root_classes.size() != Y_base_classes.size():
            root_left = root_left.left_child
            root_classes, _ = torch.sort(torch.unique(root_left.classes))

        # create root node and assign child nodes to it
        root = Node_Gibbs()
        self.root = root

        self.root.set_child(root_left, 0)
        self.root.set_child(root_right, 1)

        # classes of current node
        map_classes_left = {i: 0 for i in detach_to_numpy(torch.unique(Y_base)).tolist()}
        map_classes_right = {i: 1 for i in detach_to_numpy(torch.unique(Y_novel)).tolist()}
        root_old_to_new = {**map_classes_left, **map_classes_right}

        Y_root, _ = torch.sort(torch.cat((root_left.classes, root_right.classes)))

        self.root.set_data(Y_root, root_old_to_new)

        # the data for novel classes is X,Y while the data for base classes is the Xbars
        gp_counter += 1
        num_data = Y_base.shape[0] + Y_novel.shape[0]
        self.root.set_model(self.args.kernel_function, self.args.num_steps, self.args.num_draws,
                            self.args.gibbs_outputscale, self.args.gibbs_lengthscale, num_data)

        self._adjust_node_info()
        return gp_counter

    def _adjust_node_info(self):
        # root
        q = deque()
        curr_id = 1

        self.root.id = 0
        self.root.depth = 0

        # push source vertex into the queue
        q.append(self.root)
        while q:
            root = q.popleft()
            # leaf node
            if root.left_child is None and root.right_child is None:
                continue
            # Internal node
            else:
                left_child = root.left_child
                right_child = root.right_child

                left_child.id = curr_id
                curr_id += 1
                left_child.depth = root.depth + 1
                q.append(left_child)

                right_child.id = curr_id
                curr_id += 1
                right_child.depth = root.depth + 1
                q.append(right_child)

    def print_tree(self):
        if self.root is not None:
            self._print_tree(self.root)

    def _print_tree(self, node):
        if node is not None:
            self._print_tree(node.left_child)
            logging.info(str(node.classes) + ' ')
            self._print_tree(node.right_child)

    def train_tree(self, X, Y, X_idx, Xbar, to_print=True):
        loss = 0
        if self.root is not None:
            loss = self._train_tree(self.root, X, Y, X_idx, Xbar, to_print)
        return loss

    def _train_tree(self, node, X, Y, X_idx, Xbar, to_print=True):

        loss = 0
        node_classes = set(detach_to_numpy(node.classes).tolist())
        batch_classes = set(detach_to_numpy(Y).tolist())

        # enter if it is an internal node and there are at least 1 example in the batch for that node
        if node.classes.size(0) > 1 and len(batch_classes.intersection(node_classes)) > 0:
            loss += self._train_tree(node.left_child, X, Y, X_idx, Xbar, to_print)
            if to_print:
                logging.info('Training GP on classes: ' + str(detach_to_numpy(node.classes).tolist()) + ' ')
            node_X, node_Y, node_idx, node_Xbar = self._extract_node_data(node, X, Y, X_idx, Xbar)
            loss += node.train_loop(node_X, node_Y, node_idx, node_Xbar, to_print)
            loss += self._train_tree(node.right_child, X, Y, X_idx, Xbar, to_print)
        else:
            if to_print:
                logging.info('No need for training. Class: ' + str(detach_to_numpy(node.classes).tolist()) + ' ')
        return loss

    def _extract_node_data(self, node, X, Y, X_idx, Xbar):
        # take data belongs to that node only
        node_X, node_Y_orig, node_Xidx_orig = pytorch_take2(X, Y, X_idx, node.classes)
        # from original labels to node labels
        node_Y = torch.tensor([node.old_to_new[y.item()] for y in node_Y_orig], dtype=Y.dtype).to(Y.device)
        # extract indices of batch examples
        node_idx = torch.tensor([node.node_orig_to_new_idx[i.item()] for i in node_Xidx_orig]).long().to(Y.device)
        node_Xbar = Xbar[node.classes, ...]
        return node_X, node_Y, node_idx, node_Xbar

    def eval_tree_full_path(self, X, num_classes, Xbar, jitter_Kmm=False):
        # create a queue used to do BFS
        q = deque()

        # accumulated log probability matrix
        probs_mat = torch.ones((X.shape[0], num_classes), dtype=X.dtype, device=X.device)

        # push source vertex into the queue
        q.append(self.root)

        # loop till queue is empty
        while q:
            # pop front node from queue and print it
            node = q.popleft()

            # In case of only one class, all predictions are of that class. Nothing to add to the queue
            if node.classes.size(0) == 1:
                #logging.info('No need for evaluation. Class: ' + str(node.classes) + ' ')
                continue

            # In case more than one class run GP on the node
            else:
                #logging.info('Evaluating GP on classes: ' + str(node.classes) + ' ')

                Xbar_r = Xbar[node.classes, ...]
                X_star = torch.cat((Xbar_r.reshape(-1, Xbar_r.shape[-1]), X), dim=0)

                probs = node.model.predictive_posterior(X_star, jitter_Kmm=jitter_Kmm)
                left_classes = node.new_to_old[0]
                right_classes = node.new_to_old[1]

                probs = probs.unsqueeze(1)
                class_probs = torch.cat((1 - probs, probs), dim=1)

                probs_mat[:, left_classes] = probs_mat[:, left_classes] * class_probs[:, 0].reshape(-1, 1)
                probs_mat[:, right_classes] = probs_mat[:, right_classes] * class_probs[:, 1].reshape(-1, 1)

                # more than 2 childes - not a leaf node. Add child nodes to queue.
                if node.classes.size(0) > 2:
                    q.append(node.left_child)
                    q.append(node.right_child)

        return probs_mat

    def build_tree_gibbs(self, root, X, Y):
        """
        Build binary tree with GP attached to each node
        """
        # root
        q = deque()

        # push source vertex into the queue
        q.append((root, X, Y))
        curr_id = 1
        gp_counter = 0  # for getting avg. loss over the whole tree

        # loop till queue is empty
        while q:
            # pop front node from queue
            root, root_X, root_Y = q.popleft()
            node_classes, _ = torch.sort(torch.unique(root_Y))
            num_classes = node_classes.size(0)

            # two classes or less - no heuristic for splitting
            split_method = 'MeanSplitKmeans' if num_classes > 2 else 'Split'
            root_old_to_new = \
                self.split_func(detach_to_numpy(root_X),
                                detach_to_numpy(root_Y))[split_method].split()

            root.set_data(root_Y, root_old_to_new)

            # leaf node
            if num_classes == 1:
                logging.info('Reached a leaf node. Node index: ' + str(root.id) + ' ')
                continue

            # Internal node
            else:
                gp_counter += 1
                root.set_model(self.args.kernel_function, self.args.num_steps, self.args.num_draws,
                               self.args.gibbs_outputscale, self.args.gibbs_lengthscale, root_X.shape[0])

                left_X, left_Y = pytorch_take(root_X, root_Y, root.new_to_old[0])
                right_X, right_Y = pytorch_take(root_X, root_Y, root.new_to_old[1])
                child_X = [left_X, right_X]
                child_Y = [left_Y, right_Y]

                branches = 2
                for i in range(branches):
                    child = Node_Gibbs()
                    child.id = curr_id
                    curr_id += 1
                    child.depth = root.depth + 1
                    root.set_child(child, i)
                    q.append((child, child_X[i], child_Y[i]))

        return gp_counter

    def create_root(self):
        self.root = Node_Gibbs()

    def train_right_subtree(self, X, Y, to_print=True):
        """
        Training on the root node (with new data and inducing points of previous data)
        and the right sub tree only with the new data
        :param X: [left_tree_Xbar, right_tree_X]
        :param Y: [left_tree_Xbar_y, right_tree_Y]
        :param X_idx: [Xbar_idx, right_tree_Y]
        :return: loss overall tree
        """
        loss = 0
        if self.root is not None:
            # train on root node
            if to_print:
                logging.info('Training GP on classes: ' + str(detach_to_numpy(self.root.classes).tolist()) + ' ')
            node_X, node_Y = self._gibbs_extract_node_data(self.root, X, Y)
            loss += self.root.train_loop(node_X, node_Y, to_print)
            # train only right subtree
            loss += self._gibbs_train_tree(self.root.right_child, X, Y, to_print)
        return loss

    def _gibbs_train_tree(self, node, X, Y, to_print=True):
        loss = 0
        node_classes = set(detach_to_numpy(node.classes).tolist())
        batch_classes = set(detach_to_numpy(Y).tolist())

        # enter if it is an internal node and there are at least 1 example in the batch for that node
        if node.classes.size(0) > 1 and len(batch_classes.intersection(node_classes)) > 0:
            loss += self._gibbs_train_tree(node.left_child, X, Y, to_print)
            if to_print:
                logging.info('Training GP on classes: ' + str(detach_to_numpy(node.classes).tolist()) + ' ')
            node_X, node_Y = self._gibbs_extract_node_data(node, X, Y)
            loss += node.train_loop(node_X, node_Y, to_print)
            loss += self._gibbs_train_tree(node.right_child, X, Y, to_print)
        else:
            if to_print:
                logging.info('No need for training. Class: ' + str(detach_to_numpy(node.classes).tolist()) + ' ')
        return loss

    def _gibbs_extract_node_data(self, node, X, Y):
        # take data belongs to that node only
        node_X, node_Y_orig = pytorch_take(X, Y, node.classes)
        # from original labels to node labels
        node_Y = torch.tensor([node.old_to_new[y.item()] for y in node_Y_orig], dtype=Y.dtype).to(Y.device)
        return node_X, node_Y

    def gibbs_eval_tree_full_path(self, X, num_classes, Xbar):
        # create a queue used to do BFS
        q = deque()

        # accumulated log probability matrix
        probs_mat = torch.ones((X.shape[0], num_classes), dtype=X.dtype, device=X.device)

        # push source vertex into the queue
        q.append(self.root)

        # loop till queue is empty
        while q:
            # pop front node from queue and print it
            node = q.popleft()

            # In case of only one class, all predictions are of that class. Nothing to add to the queue
            if node.classes.size(0) == 1:
                #logging.info('No need for evaluation. Class: ' + str(node.classes) + ' ')
                continue

            # In case more than one class run GP on the node
            else:
                #logging.info('Evaluating GP on classes: ' + str(node.classes) + ' ')
                if isinstance(node, Node_VI):
                    Xbar_r = Xbar[node.classes, ...]
                    X_star = torch.cat((Xbar_r.reshape(-1, Xbar_r.shape[-1]), X), dim=0)
                else:
                    X_star = X

                probs = node.model.predictive_posterior(X_star)
                left_classes = node.new_to_old[0]
                right_classes = node.new_to_old[1]

                probs = probs.unsqueeze(1)
                class_probs = torch.cat((1 - probs, probs), dim=1)

                probs_mat[:, left_classes] = probs_mat[:, left_classes] * class_probs[:, 0].reshape(-1, 1)
                probs_mat[:, right_classes] = probs_mat[:, right_classes] * class_probs[:, 1].reshape(-1, 1)

                # more than 2 childes - not a leaf node. Add child nodes to queue.
                if node.classes.size(0) > 2:
                    q.append(node.left_child)
                    q.append(node.right_child)

        return probs_mat