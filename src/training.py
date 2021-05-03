import numpy as np
import torch
from time import time
from torch.autograd import Variable
from src.datasets import triplet_dataloader
from src.utils import *

def prep_triplets(triplets, cuda):
    """
    Takes a batch of triplets and converts them into Pytorch variables 
    and puts them on GPU if available.
    """
    a, n, d = (Variable(triplets['anchor']), Variable(triplets['neighbor']), Variable(triplets['distant']))
    if cuda:
    	a, n, d = (a.cuda(), n.cuda(), d.cuda())
    return (a, n, d)

'''
def prep_triplets_noise(triplets, noise, cuda):
    a, n, d, noise = (Variable(triplets['anchor']), Variable(triplets['neighbor']), 
                      Variable(triplets['distant']), Variable(noise))
    if cuda:
        a, n, d, noise = (a.cuda(), n.cuda(), d.cuda(), noise.cuda())

    return (a, n, d, noise)
'''

# +
def adjust_embedding(z_a, z_n, z_d, sampling=False):
    if sampling:
        batch_size = z_a.shape[0]

        za_unsq = torch.unsqueeze(z_a, 1)
        zn_unsq = torch.unsqueeze(z_n, 1)
        zd_unsq = torch.unsqueeze(z_d, 1)
        raw_z = torch.cat((za_unsq, zn_unsq, zd_unsq), 1)
        
        index = torch.randint(0, 3, (batch_size, )).view(-1, 1, 1).expand(raw_z.size(0), 1, raw_z.size(2))
        
        cuda = torch.cuda.is_available() 
        
        if cuda:
            index = index.cuda()
        
        z = raw_z.gather(1, index).view(batch_size, -1)
    else:
        z = torch.cat((z_a, z_n, z_d))
        
    return z

def train_triplet_noise_epoch(model, cuda, dataloader, 
                              dataset, batch_size, optimizer, scheduler, 
                              epoch, ut=3, margin=1, alpha=0, 
                              print_every=100, t0=None, sampling=False):

    model.train()

    update_targets = bool((epoch + 1) % ut == 0)
    model.train(True)

    if t0 is None:
            t0 = time.time()

    sum_loss, sum_l_a, sum_l_n, sum_l_d, sum_noise = (0, 0, 0, 0, 0)
    n_train, n_batches = len(dataloader.dataset), len(dataloader)
    print_sum_loss = 0

    for batch_idx, (idx, triplets, noise)  in enumerate(dataloader):

        if cuda:
            a, n, d = prep_triplets(triplets, cuda)
            
        optimizer.zero_grad()
        
        z_a, z_n, z_d = (model.encode(a), model.encode(n), model.encode(d))
        
        z = adjust_embedding(z_a, z_n, z_d, sampling=sampling)
        
        if sampling:
            noise = noise[:, 0 , :]
        else:
            anchor_noise = noise[:, 0, :]
            neighbor_noise = noise[:, 1, :]
            distant_noise = noise[:, 2, :]
            noise = torch.cat((anchor_noise, neighbor_noise, distant_noise))

        if update_targets:
            e_targets = noise.numpy()
            e_out = z.cpu().data.numpy()
            new_targets = calc_optimal_target_permutation(e_out, e_targets)
            dataset.update_targets(idx, new_targets, sampling=sampling)
            noise = torch.FloatTensor(new_targets)
        
        if cuda:
            noise = noise.cuda()
        
        noise = Variable(noise)
        
        loss_noise = model.noise_loss(z, noise, alpha)

        loss, l_a, l_n, l_d = model.loss(a, n, d, margin=margin)
        
        loss += loss_noise

        loss.backward()
        optimizer.step()

        sum_loss += loss.data.item()
        sum_l_a += l_a.data.item()
        sum_l_n += l_n.data.item()
        sum_l_d += l_d.data.item()
        sum_noise += loss_noise.item()

        if (batch_idx + 1) * dataloader.batch_size % print_every == 0:
            print_avg_loss = (sum_loss - print_sum_loss) / (print_every / dataloader.batch_size)
            print('Epoch {}: [{}/{} ({:0.0f}%)], Avg loss: {:0.4f}'.format(
                epoch, (batch_idx + 1) * dataloader.batch_size, n_train,
                100 * (batch_idx + 1) / n_batches, print_avg_loss))
            print_sum_loss = sum_loss
    
    scheduler.step()

    avg_loss = sum_loss / n_batches
    avg_l_a = sum_l_a / n_batches
    avg_l_n = sum_l_n / n_batches
    avg_l_d = sum_l_d / n_batches
    avg_noise = sum_noise / n_batches

    print('Finished epoch {}: {:0.3f}s'.format(epoch, time()-t0))
    print('  Average loss: {:0.4f}'.format(avg_loss))
    print('  Average l_a: {:0.4f}'.format(avg_l_a))
    print('  Average l_n: {:0.4f}'.format(avg_l_n))
    print('  Average l_d: {:0.4f}'.format(avg_l_d))
    print('  Average noise: {:0.4f}\n'.format(avg_noise))

    return (avg_loss, avg_l_a, avg_l_n, avg_l_d, avg_noise)


# -

def train_triplet_epoch(model, cuda, dataloader, optimizer, epoch, margin=1,
    l2=0, print_every=100, t0=None):
    """
    Trains a model for one epoch using the provided dataloader.
    """
    model.train()
    if t0 is None:
        t0 = time.time()
    
    sum_loss, sum_l_n, sum_l_d, sum_l_nd = (0, 0, 0, 0)
    n_train, n_batches = len(dataloader.dataset), len(dataloader)
    print_sum_loss = 0
    
    for idx, triplets in enumerate(dataloader):
        p, n, d = prep_triplets(triplets, cuda)
        
        optimizer.zero_grad()
        
        loss, l_n, l_d, l_nd = model.loss(p, n, d, margin=margin, l2=l2)
        loss.backward()
        optimizer.step()
        
        sum_loss += loss.data.item()
        sum_l_n += l_n.data.item()
        sum_l_d += l_d.data.item()
        sum_l_nd += l_nd.item()
        
        if (idx + 1) * dataloader.batch_size % print_every == 0:
            print_avg_loss = (sum_loss - print_sum_loss) / (
                print_every / dataloader.batch_size)
            print('Epoch {}: [{}/{} ({:0.0f}%)], Avg loss: {:0.4f}'.format(
                epoch, (idx + 1) * dataloader.batch_size, n_train,
                100 * (idx + 1) / n_batches, print_avg_loss))
            print_sum_loss = sum_loss
    
    avg_loss = sum_loss / n_batches
    avg_l_n = sum_l_n / n_batches
    avg_l_d = sum_l_d / n_batches
    avg_l_nd = sum_l_nd / n_batches
    
    print('Finished epoch {}: {:0.3f}s'.format(epoch, time()-t0))
    print('  Average loss: {:0.4f}'.format(avg_loss))
    print('  Average l_n: {:0.4f}'.format(avg_l_n))
    print('  Average l_d: {:0.4f}'.format(avg_l_d))
    print('  Average l_nd: {:0.4f}\n'.format(avg_l_nd))
    
    return (avg_loss, avg_l_n, avg_l_d, avg_l_nd)
