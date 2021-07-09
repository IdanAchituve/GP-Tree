from torchsummary import summary

import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import trange
from utils import *
from torchvision import transforms
from FSCIL.cub.dataloader import CUB200_Indexed
from GP_Tree.Learner import ModelBinaryTree
from io import BytesIO

torch.set_printoptions(profile="full")

parser = argparse.ArgumentParser(description='FSCIL GP - trainer')
parser.add_argument('--script-name', default='CUB')
parser.add_argument('--exp-name', type=str, default='', metavar='N',
                    help='experiment name suffix')
parser.add_argument('--num-sessions', type=int, default=10, help='Number of few shot sessions')
parser.add_argument('--N-way', type=lambda s: [int(item.strip()) for item in s.split(',')],
                    default='100,10,10,10,10,10,10,10,10,10,10',
                    help='number of classes per session')
parser.add_argument('--N-shot', type=lambda s: [int(item.strip()) for item in s.split(',')],
                    default='10000000,5,5,5,5,5,5,5,5,5,5',
                    help='Number of samples per session')
parser.add_argument('--NN-layers', type=lambda s: [int(item.strip()) for item in s.split(',')],
                    default='512',
                    help='layers after feature extractor')
parser.add_argument('--base-num-epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('--move-to-gp-epoch', type=int, default=20, help='epoch to start training with GP')
parser.add_argument('--dataroot', type=str, default='./dataset', help='dataset root')
parser.add_argument('--scheduler', default=True, type=str2bool, help='use learning rate scheduler')
parser.add_argument('--optimizer', default='sgd', choices=['adam', 'sgd'], type=str,
                    help='use learning rate scheduler')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum value for optimizer, default is 0.9.')
parser.add_argument('--base-lr', default=1e-2, type=float, help='learning rate')
parser.add_argument('--natural-lr', default=.1, type=float,
                    help='natural GA learning rate. If not using stochastic updates - may use a value of 1.')
parser.add_argument('--wd', default=1e-4, type=float, help='weight decay')
parser.add_argument('--batch-size', type=int, default=128, help='batch size')
parser.add_argument('--test-batch-size', type=int, default=256, help='test batch size')
parser.add_argument('--base-milestones', type=lambda s: [int(item.strip()) for item in s.split(',')],
                    default='40,60')
parser.add_argument('--num-steps', type=int, default=10, help='number of sampling iterations')
parser.add_argument('--num-draws', type=int, default=30, help='number of parallel gibbs chains')
parser.add_argument('--kernel-function', type=str, default='RBFKernel',
                    choices=['RBFKernel', 'LinearKernel', 'MaternKernel'],
                    help='kernel function')
parser.add_argument('--num-inducing-points', type=int, default=5,
                    help='Number of inducing points per class')
parser.add_argument('--learn-location', default=True, type=str2bool, help='learn inducing point location')
parser.add_argument('--gibbs-outputscale', type=float, default=8., help='output scale')
parser.add_argument('--gibbs-lengthscale', type=float, default=1., help='length scale')
parser.add_argument('--outputscale', type=float, default=4., help='output scale')
parser.add_argument('--lengthscale', type=float, default=1., help='length scale')
parser.add_argument('--eval-every', type=int, default=10, help='num. epochs between test set eval')
parser.add_argument('--out-dir', type=str, default='./outputs', help='Output dir')
parser.add_argument('--seed', default=42, type=int, help='random seed')
parser.add_argument('--num-workers', default=4, type=int, help='num wortkers')
parser.add_argument('--gpus', type=str, default='0',
                    help='comma delimited of gpu ids to use. Use "-1" for cpu usage')

args = parser.parse_args()

# logger config
set_logger()

exp_name = f'FSCIL_CUB_seed_{args.seed}_outputscale_{args.outputscale}_gibbs_outputscale_{args.gibbs_outputscale}' \
           f'base_lr_{args.base_lr}_kernel_func_{args.kernel_function}_natural_lr_{args.natural_lr}'

if args.exp_name != '':
    exp_name += '_' + args.exp_name

logging.info(str(args))
device = get_device(cuda=int(args.gpus) >= 0, gpus=args.gpus)

# =================
# get loaders
# =================
batch_size = args.batch_size
test_batch_size = args.test_batch_size
N_way = args.N_way
N_shot = args.N_shot
dataset_path = args.dataroot

transform_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


def create_data_loaders():
    train_loaders = []
    test_loaders = []

    # num_sessions counts the novel session. Here need base + novel
    num_sessions = args.num_sessions + 1
    for i in range(1, num_sessions + 1):
        trainset_novel = CUB200_Indexed(logger=logging, root=dataset_path, dataset='train',
                                        transform=transform_train, session=i, val_indices_path='val_indices')
        # drop last needs to be False to accommodate for few shot sessions
        train_loaders.append(DataLoader(trainset_novel, batch_size=batch_size, num_workers=args.num_workers,
                                  shuffle=True, drop_last=False))

        # validation loader only for base classes
        if i == 1:
            valset_novel = CUB200_Indexed(logger=logging, root=dataset_path, dataset='val',
                                          transform=transform_test, session=i, val_indices_path='val_indices')
            base_val_loader = DataLoader(valset_novel, batch_size=test_batch_size, num_workers=args.num_workers)

        testset_novel = CUB200_Indexed(logger=logging, root=dataset_path, dataset='test',
                                       transform=transform_test, session=i)
        test_loaders.append(DataLoader(testset_novel, batch_size=test_batch_size, num_workers=args.num_workers))

    return train_loaders, base_val_loader, test_loaders

train_dataloader, base_val_loader, test_dataloader = create_data_loaders()

# set seed here after data split
set_seed(args.seed)

# =================
# Model
# =================
gp_counter = 0

# build initial model
model = ModelBinaryTree(args, device)

# print summary
summary(model.features, input_size=(3, 224, 224), device='cpu')
model.to(device)

dir_name = 'gp_tree_output'
args.out_dir = (Path(args.out_dir) / dir_name).as_posix()
out_dir = save_experiment(args, None, return_out_dir=True, save_results=False)
logging.info(out_dir)


def model_save(model, file=None):
    if file is None:
        file = BytesIO()
    torch.save({'model_state_dict': model.state_dict()}, file)
    return file


def model_load(file):
    if isinstance(file, BytesIO):
        file.seek(0)

    model.load_state_dict(
        torch.load(file, map_location=lambda storage, location: storage)['model_state_dict']
    )

    return model

# ==========
# optimizers
# ==========
optimizer = optim.SGD(model.parameters(), lr=args.base_lr, weight_decay=args.wd, momentum=args.momentum) \
    if args.optimizer == 'sgd' else optim.Adam(model.parameters(), lr=args.base_lr, weight_decay=args.wd)

if args.scheduler:
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.base_milestones, gamma=0.1)


# ================
# Evaluate
# ================
# evaluating test/val data
def model_evalution(loader, num_classes, gibbs=False):

    model.eval()
    targets = []
    preds = []

    cumm_loss = 0
    num_samples = 0

    with torch.no_grad():
        for k, batch in enumerate(loader):
            batch = (t.to(device) for t in batch)
            data, clf_labels, indices = batch
            loss, pred = model.forward_eval(data, clf_labels, num_classes) if not gibbs else \
                         model.gibbs_forward_eval(data, clf_labels, num_classes)

            targets.append(clf_labels)
            preds.append(pred)

            num_samples += clf_labels.shape[0]
            cumm_loss += loss.item() * clf_labels.shape[0]

        cumm_loss /= num_samples
        target = detach_to_numpy(torch.cat(targets, dim=0))
        full_pred = detach_to_numpy(torch.cat(preds, dim=0))
        pred_top1 = topk(target, full_pred, 1)
        pred_top3 = pred_top5 = 1.
        if num_classes >= 3:
            pred_top3 = topk(target, full_pred, 3)
        if num_classes >= 5:
            pred_top5 = topk(target, full_pred, 5)

    return (pred_top1, pred_top3, pred_top5), cumm_loss


# ==========
# Base Train
# ==========
def save_data(tensor, file):
    torch.save(tensor, file)


def load_data(file):
    return torch.load(file, map_location=lambda storage, location: storage)


def base_logging(epoch, cumm_loss, val_loss, val_accuracies, test_loss, test_accuracies):
    logging.info(f"Base Epoch: {epoch}, "
                 f"Base Train Loss: {cumm_loss:.5f}, "
                 f"Base Val Loss: {val_loss:.5f}, "
                 f"Base Val accuracy: {val_accuracies[0]:.5f}, "
                 f"Base Val accuracy top3: {val_accuracies[1]:.5f}, "
                 f"Base Val accuracy top5: {val_accuracies[2]:.5f}, "
                 f"Base Test Loss: {test_loss:.5f}, "
                 f"Base Test accuracy: {test_accuracies[0]:.5f}, "
                 f"Base Test accuracy top3: {test_accuracies[1]:.5f}, "
                 f"Base Test accuracy top5: {test_accuracies[2]:.5f}")


base_train_loader = train_dataloader[0]
base_test_loader = test_dataloader[0]
epoch_iter = trange(args.base_num_epochs)
test_accuracies = (0, 0, 0)
val_accuracies = (0, 0, 0)
test_loss = val_loss = 0
built_tree = False
base_classes = torch.arange(N_way[0]).to(device)

for epoch in epoch_iter:
    best_loss = best_test_loss = np.inf
    best_test_labels_vs_preds = 0
    cumm_loss = num_samples = 0

    model.train()
    to_print = True
    for k, batch in enumerate(base_train_loader):

        batch = (t.to(device) for t in batch)
        train_data, clf_labels, indices = batch

        optimizer.zero_grad()
        loss = model(train_data, clf_labels, indices, to_print=to_print)

        loss.backward()
        optimizer.step()

        num_samples += clf_labels.shape[0]
        epoch_iter.set_description(f'[{epoch} {k}] Training loss {loss.item():.5f}')
        cumm_loss += loss.item() * clf_labels.shape[0]
        to_print = False

        # save all samples/labels/indices
        if epoch == args.move_to_gp_epoch or \
                (args.move_to_gp_epoch > (args.base_num_epochs - 1) and epoch == (args.base_num_epochs - 1)):
            with torch.no_grad():
                z = model.get_features(train_data)
                X = torch.cat((X, z), dim=0) if k > 0 else z
                Y = torch.cat((Y, clf_labels), dim=0) if k > 0 else clf_labels
                X_idx = torch.cat((X_idx, indices), dim=0) if k > 0 else indices

    cumm_loss /= num_samples
    logging.info('-------------------- Test - Epoch ' + str(epoch) + '---------------------')
    if (epoch + 1) % args.eval_every == 0 or epoch == args.base_num_epochs - 1:
        val_accuracies, val_loss = model_evalution(base_val_loader, N_way[0])
        test_accuracies, test_loss = model_evalution(base_test_loader, N_way[0])

    # log
    base_logging(epoch, cumm_loss, val_loss, val_accuracies, test_loss, test_accuracies)

    if args.scheduler:
        scheduler.step()
        lrs = scheduler.get_last_lr()
        logging.info(f"learning rate is {lrs[0]:.5f}\n")

    # build tree
    if epoch == args.move_to_gp_epoch or (epoch == args.base_num_epochs - 1 and not built_tree):
        logging.info('--------------------Training GP: ' + str(epoch) + ' ---------------------')
        save_data(X, out_dir / "X.pt")
        save_data(Y, out_dir / "Y.pt")
        save_data(X_idx, out_dir / "X_idx.pt")

        with torch.no_grad():
            gp_counter += model.build_base_tree(X, Y, X_idx)

        # add GP params to optimizer
        params = {'params': (p for n, p in model.named_parameters()
                             if 'outputscale' in n or 'lengthscale' in n)}
        optimizer.add_param_group(params)
        built_tree = True

# =============
# Novel Training
# =============
def expand_tree(X_old, Y_old, X_new, Y_new):
    gp_counter_novel = model.expend_tree(X_old, Y_old, X_new, Y_new)
    return gp_counter_novel


def log_metrics(epoch, session, cumm_loss, test_loss, test_accuracies):
    logging.info(f"Session {session} Epoch: {epoch}, "
                 f"Session {session} Train Loss: {cumm_loss:.5f}, "
                 f"Session {session} Test Loss: {test_loss:.5f}, "
                 f"Session {session} Test accuracy: {test_accuracies[0]:.5f}, "
                 f"Session {session} Test accuracy top3: {test_accuracies[1]:.5f}, "
                 f"Session {session} Test accuracy top5: {test_accuracies[2]:.5f}")


for session in range(1, args.num_sessions + 1):
    logging.info('==============================================================================')
    logging.info('-------------------- Training Session' + str(session) + '---------------------')
    logging.info('==============================================================================')

    sess_train_loader = train_dataloader[session]
    sess_test_loader = test_dataloader[session]

    # get "examplers" of previous classes - should be ordered according to class number
    num_prev_classes = sum(N_way[:session])
    previous_classes = torch.arange(num_prev_classes).to(device)

    with torch.no_grad():
        num_prev_novel_classes = sum(N_way[1:session])
        previous_novel_classes = torch.arange(N_way[0], N_way[0] + num_prev_novel_classes).to(device)
        X_prev_novel = None
        Y_prev_novel = None
        if num_prev_novel_classes > 0:
            X_prev_novel = model.Xbar[previous_novel_classes, ...].clone().detach().flatten(start_dim=0, end_dim=1)
            Y_prev_novel = previous_novel_classes.clone().long().reshape(-1, 1). \
                           expand((num_prev_novel_classes, args.num_inducing_points)).reshape(-1).to(device)

        X_left = model.Xbar[base_classes, ...].clone().detach().flatten(start_dim=0, end_dim=1)
        Y_left = base_classes.clone().long().reshape(-1, 1). \
                 expand((N_way[0], args.num_inducing_points)).reshape(-1).to(device)

        test_loss = 0
        test_accuracies = (0, 0, 0)
        novel_classes = torch.arange(num_prev_classes, num_prev_classes + N_way[session]).to(device)

        cumm_loss = num_samples = 0

        model.train()
        model.disable_bn()  # disable BN after base training
        to_print = True

        for k, batch in enumerate(sess_train_loader):

            batch = (t.to(device) for t in batch)
            train_data, clf_labels, indices = batch
            orig_clf_labels = clf_labels.clone()

            train_data = (X_prev_novel, train_data)
            clf_labels = (Y_prev_novel, clf_labels)

            # expend tree with novel data. counting on that each batch contains all the data
            gp_counter_novel = expand_tree(X_left, Y_left, train_data, clf_labels)

            loss = model.forward_gibbs(X_left, Y_left, train_data, clf_labels, to_print=to_print)
            num_samples += orig_clf_labels.shape[0]
            cumm_loss += loss.item() * orig_clf_labels.shape[0]

        cumm_loss /= num_samples
        logging.info('-------------------- Test - Session ' + str(session) + '---------------------')
        test_accuracies, test_loss = model_evalution(sess_test_loader, num_prev_classes + N_way[session], gibbs=True)

        log_metrics(0, session, cumm_loss, test_loss, test_accuracies)