import numpy as np
import math
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from contextlib import nullcontext
import json




class Hessian(object):
    def __init__(self, model = None,  m = 100, sigma = 1e-5**0.5, ckpt_iteration= 0, train_data = [], block_size = None, batch_size = None, num_v = 10, ctx =nullcontext(), use_minibatch = True, gradient_accumulation_steps = 1, device = 'cuda',  sample_layer = None, ddp = False, comment = None):
        self.model = model
        self.m = m # number of lanzcos basis
        self.sigma = sigma # the standard deviation of gaussian r.v.
        self.ckpt_iteration = ckpt_iteration
        self.train_data = train_data
        self.block_size = block_size
        self.batch_size = batch_size
        self.ctx = ctx
        self.use_minibatch = use_minibatch
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.device = device
        self.sample_layer = sample_layer
        self.ddp = ddp
        self.num_v = num_v
        self.num_bins = 1000


 
        total_elements = len(self.train_data)
        self.num_batches = total_elements // (self.batch_size * self.block_size)
        
        print('total batch', self.num_batches)

        self.total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        #n_params = sum(p.numel() for p in self.parameters())
        print('total params', self.total_params)
   

        self.comment = comment + '_minibatch_'+str(self.use_minibatch) +'_bs_'+str(self.batch_size*(self.gradient_accumulation_steps))+ '_m_'+str(self.m)  + '_v_' +str(self.num_v) + '_ckpt_'+str(self.ckpt_iteration)

        self.file_dir = 'files/'+str(self.comment)+'/'

        os.makedirs(self.file_dir, exist_ok= True)


    def get_spectrum(self, layer_by_layer = False):
        if layer_by_layer: 
            self.get_spectrum_layer_by_layer()
        else: 
            self.get_spectrum_full()


    def get_spectrum_layer_by_layer(self):
      
        weights_dic, values_dic = {}, {}

        for name, param in self.model.named_parameters():
            if name not in self.sample_layer:
                continue
            if param.requires_grad:

                zeros = np.zeros((self.num_v, self.m))
                weights_dic[name] = [row.tolist() for row in zeros]
                values_dic[name] =  [row.tolist() for row in zeros]

    
        t_s = time.time()
        for k in range(self.num_v): 
            print('current k' , k)

            'wiki version'
            T_dic = self.tridiagonalize_by_lanzcos_layer_by_layer(k) #returns a dic: {'name': T}
            
            for name, T in T_dic.items():
                T = np.ones((self.m, self.m))
                eigenvalues, U  = np.linalg.eigh(T)
                values_dic[name][k] = eigenvalues.tolist() #array to list
                weights_dic[name][k] = (U[0]**2).tolist()


            'we also save the inter-medium results'
            self.save_curve(total_time= time.time() - t_s, weights_layer = weights_dic, values_layer = values_dic)



        total_time = time.time() - t_s

        self.save_curve(total_time= total_time, weights_layer = weights_dic, values_layer = values_dic)



    def get_spectrum_full(self):
      
        weights = np.zeros((self.num_v, self.m))
        values = np.zeros((self.num_v, self.m))
        time_initial = time.time()

        for k in range(self.num_v): 
            'wiki version'
            T = self.tridiagonalize_by_lanzcos(k)
            eigenvalues, U  = np.linalg.eigh(T)
            values[k,:] = eigenvalues
            weights[k,:] = U[0]**2
   

            self.save_curve(total_time = time.time() -time_initial, weights_full =  {'weights': weights}, values_full = {'values': values}, grid = [], curve = [])
            
        total_time = time.time() -time_initial
        grid, curve = self.interpolate(weights, values)

        self.save_curve(total_time = total_time, weights_full =  {'weights': weights}, values_full = {'values': values}, grid = grid, curve = curve)

    def save_curve(self,total_time = None, weights_layer = None, values_layer = None, weights_full = None, values_full = None, grid = [], curve = []):

        if total_time != None:         
            file_name = self.file_dir + 'time.txt'
            with open(file_name, "w") as file:
                file.write(str(total_time) + "\n")

        if weights_layer != None:
            weights_layer = {key: weights_layer[key] for key in weights_layer} # convert the values to list
            file_name = self.file_dir + 'weights_layer.json'
            with open(file_name, 'w') as json_file:
                json.dump(weights_layer, json_file)
        

        if values_layer != None:
            values_layer = {key: values_layer[key] for key in values_layer} # convert the values to list
            file_name = self.file_dir + 'values_layer.json'
            with open(file_name, 'w') as json_file:
                json.dump(values_layer, json_file)

        if weights_full != None:
            weights_full = {key: weights_full[key].tolist() for key in weights_full} # convert the values to list
            file_name = self.file_dir + 'weights_full.json'
            with open(file_name, 'w') as json_file:
                json.dump(weights_full, json_file)
        

        if values_full != None:
            values_full = {key: values_full[key].tolist() for key in values_full} # convert the values to list
            file_name = self.file_dir + 'values_full.json'
            with open(file_name, 'w') as json_file:
                json.dump(values_full, json_file)


        if len(grid) != 0: 
            file_name = self.file_dir+ 'grid.txt'
            with open(file_name, "w") as file:
                for item in grid:
                    file.write(str(item) + "\n")

        if len(curve) != 0:
            file_name =  self.file_dir + 'curve.txt'
            with open(file_name, "w") as file:
                for item in curve:
                    file.write(str(item) + "\n")
        

    def load_curve(self, layer_by_layer = False):
        if layer_by_layer: 
            self.load_curve_layer_by_layer()
        else: 
            self.load_curve_full()

      
    def load_curve_layer_by_layer(self):

        'load weights and values:'
        file_name = self.file_dir + 'weights_layer.json'
        with open(file_name, 'r') as json_file:
            weights_dic = json.load(json_file)
        weights_dic = {key: np.array(value) for key, value in weights_dic.items()}


        file_name = self.file_dir + 'values_layer.json'
        with open(file_name, 'r') as json_file:
            values_dic = json.load(json_file)
        values_dic = {key: np.array(value) for key, value in values_dic.items()}



        for name in weights_dic.keys():
            weights = weights_dic[name]
            values = values_dic[name]
            grid, curve = self.interpolate(weights, values)

            print('curve',curve)
            'plot'
            plt.figure()
            plt.plot(grid, curve, label = 'approximated curve', alpha = 0.5)
            plt.xlabel('Eigenvalues')
            plt.ylabel('Frequency')
            plt.ylim([1e-10,1e2])
            plt.legend()
            plt.title(f'model at interation {self.ckpt_iteration}')
            plt.savefig(self.file_dir+'spectrum_'+name+'.png')
            plt.close()

            'log plot'
            plt.figure()
            plt.semilogy(grid, curve, label = 'approximated curve', alpha = 0.5)
            plt.xlabel('Eigenvalues')
            plt.ylabel('Frequency (log)')
            plt.ylim([1e-10,1e2])
            plt.legend()
            plt.title(f'model at interation {self.ckpt_iteration}')
            plt.savefig(self.file_dir+'/spectrum_log_'+name+'.png')
            plt.close()


    def load_curve_full(self):
        'load curve'
        grid = []
        file_name = self.file_dir + 'grid.txt'
        with open(file_name, "r") as file:
            for line in file:
                grid.append(float(line.strip()))  # Use strip() to remove 

        file_name =  self.file_dir + 'curve.txt'
        curve = []
        with open(file_name, "r") as file:
            for line in file:
                curve.append(float(line.strip()))  # Use strip() to remove 



        'plot'
        plt.figure()
        plt.plot(grid, curve, label = 'approximated curve', alpha = 0.5)
        plt.xlabel('Eigenvalues')
        plt.ylabel('Frequency')
        plt.ylim([1e-10,1e2])
        # plt.xlim([-5, 5])
        plt.legend()
        plt.title(f'model at interation {self.ckpt_iteration}')
        plt.savefig(self.file_dir+'/spectrum_full_hessian.png')
        plt.close()

        'log plot'
        plt.figure()
        plt.semilogy(grid, curve, label = 'approximated curve', alpha = 0.5)
        plt.xlabel('Eigenvalues')
        plt.ylabel('Frequency (log)')
        plt.ylim([1e-10,1e2])
        #plt.xlim([3, 5])
        plt.legend()
        plt.title(f'model at interation {self.ckpt_iteration}')
        plt.savefig(self.file_dir+'/spectrum_log_full_hessian.png')
        plt.close()


    def tridiagonalize_by_lanzcos_layer_by_layer(self, k):
        v_dic = {} # value: list
        alpha_dic = {} # value: scaler
        w_dic = {} # value: #parameters*1 tensor
        beta_dic = {} # value: scaler
        T_dic = {} # value: m*m tensor 
        'initialize'
        for name, params in self.model.named_parameters():
            if name not in self.sample_layer:
                continue
            if params.requires_grad:
                v = torch.randn_like(params, dtype = torch.float64) 
                v /= torch.norm(v)
                v_dic[name] = [v.cpu()]
                T_dic[name] = np.zeros((self.m, self.m), dtype= np.float64)


        w_prime_dic = self.hessian_vector_product_with_dic_input(v_dic, k,0) 

        'orthogonalize wprime'
        for name in T_dic.keys():
            alpha_dic[name] = torch.sum(w_prime_dic[name] * v_dic[name][-1])  
            w_dic[name] = w_prime_dic[name] - alpha_dic[name] * v_dic[name][-1]
            T_dic[name][0, 0] = alpha_dic[name] 

        'iteration'
        print('runing lanczos')
        for j in range(1, self.m):

            for name in T_dic.keys(): 
                beta = torch.norm(w_dic[name])
                beta_dic[name] = beta
                if beta >1e-8:
                    v_dic[name].append( w_dic[name] / beta )
                else:
                    #print('The value of beta is 0')
                    v_dic[name].append( w_dic[name] / 1e-8 )
                    #raise ZeroDivisionError('The value of beta is 0')
                if len(v_dic[name]) > 2:
                    del v_dic[name][0]  # keep this list short to save memory

            t_hessian = time.time()
  
            w_prime_dic = self.hessian_vector_product_with_dic_input(v_dic, k,j) 
            print('t for hessian', time.time() - t_hessian)

            'orthogonalize wprime'
            for name in T_dic.keys():
                alpha_dic[name] = torch.sum(w_prime_dic[name] * v_dic[name][-1])  
                w_dic[name] = w_prime_dic[name] - alpha_dic[name] * v_dic[name][-1] - beta_dic[name] * v_dic[name][-2]
                T_dic[name][j, j] = alpha_dic[name] 
                T_dic[name][j-1, j ] = beta_dic[name] 
                T_dic[name][j , j-1] = beta_dic[name]

        return  T_dic


    def tridiagonalize_by_lanzcos(self, k):
        'set up'
        v_list = []
        T = np.zeros((self.m, self.m), dtype= np.float64)

        'initialization'
        v = torch.randn(self.total_params, dtype = torch.float64) 
        v /= torch.norm(v)
        v_list.append(v.cpu())


        w_prime = self.hessian_vector_product_with_tensor_input(v_list[-1], k,0)
        'orthogonalize wprime'
        alpha = torch.sum(w_prime * v_list[-1])
        w = w_prime - alpha * v_list[-1]
        T[0, 0] = alpha

        'iteration'
        #t_s = time.time()
        print('runing lanczos')
        for j in range(1, self.m):
            beta = torch.norm(w)
            if beta >1e-8:
                v_list.append(w / beta)

            else:
                v_list.append(w / 1e-8)

                # print(f' since beta = {beta}, generate v that orthogonal to all previous v')
                # # Generate a random vector orthogonal to previous ones
                # v = torch.randn(self.total_params) *(1/self.total_params)**0.5
                # for i in range(j):
                #     vi = v_list[i]
                #     v -= torch.sum(vi * v) * vi
                # v /= torch.norm(v)
                if len(v_list) > 2:
                    del v_list[0]  # keep this list short to save memory


            w_prime = self.hessian_vector_product_with_tensor_input(v_list[-1], k,j)
            alpha = torch.sum(w_prime* v_list[-1])
            w = w_prime - alpha * v_list[-1] - beta * v_list[-2]
            T[j, j] = alpha
            T[j-1, j ] = beta
            T[j , j-1] = beta
         
        return  T


    def interpolate(self,weights, values):
        left_boundary = np.mean(np.min(values, axis = 1))-1
        right_boundary= np.mean(np.max(values, axis = 1)) +1
        n_grid = 50000
        grid = np.linspace(left_boundary, right_boundary, n_grid).tolist()
        density_all = np.zeros((self.num_v, n_grid))

        for k  in range(self.num_v):
            for idx, t  in enumerate(grid):
                values_each_v_t = self.gaussian_density(t, values[k,:])
                density_each_v_t = np.sum(values_each_v_t * weights[k,:])
                density_all[k,idx] = density_each_v_t

        density_avg = np.nanmean(density_all, axis = 0)
        norm_fact = np.sum(density_avg)*(grid[1]- grid[0])
        density_avg /= norm_fact

        return grid, density_avg
 
    def hessian_vector_product_with_dic_input(self, d_dic, v_step, l_step):

        'comput hessian_vector product, takes a dictionary as input, the values of dic is a list of historical lanscoz directions: d_dic = {name, [history v..]}'
        self.model.eval()
        self.model.zero_grad(set_to_none = True)

        'initialize'
        hd_dic = {}
        for name, param in self.model.named_parameters():
            if name not in self.sample_layer:
                continue
            if param.requires_grad:
                hd_dic[name]  = torch.zeros_like(param.data).cpu()


        t_hd = time.time()
        for batch_idx in range(self.num_batches):

            
            X, Y = self.get_batch(batch_idx)
            with self.ctx:
                _, loss = self.model(X, Y)

            loss.backward(create_graph= True)
            g_dic = {}
            for name, param in self.model.named_parameters():
                if name not in self.sample_layer:
                    continue
                if param.requires_grad:
                    g_dic[name] = param.grad.double()

        
            self.model.zero_grad(set_to_none = True)
            for name, param in self.model.named_parameters():
                if name not in self.sample_layer:
                    continue
                if param.requires_grad:
                    l = torch.sum(g_dic[name].cuda() * d_dic[name][-1].cuda())
                    l.backward(retain_graph = True)
                    hd = param.grad.double().data.clone()
                    hd_dic[name]  += hd.cpu()   
                    self.model.zero_grad(set_to_none = True)

            if batch_idx % 10 == 1 or batch_idx == self.gradient_accumulation_steps-1:
                print(f'layer hessian: load_iter ={self.ckpt_iteration}, current random direction = {v_step} / {self.num_v}, lanczos step = {l_step} / {self.m}, Hd current batch = {batch_idx} / {self.num_batches}, time = {time.time() -t_hd}')
                t_hd = time.time()

            if self.use_minibatch == True and batch_idx == self.gradient_accumulation_steps-1:
                break
       
        return hd_dic

    def hessian_vector_product_with_tensor_input(self, d_tensor, v_step, l_step):
        'comput hessian_vector product, takes a flattened tensors as input (with shape (total parameters, ) )'

        d_tensor = d_tensor.cuda()
        self.model.eval()
        self.model.zero_grad(set_to_none = True)
        total_hd_tensor = 0

        t_hd = time.time()
        for batch_idx in range(self.num_batches):

            X, Y = self.get_batch(batch_idx)
            with self.ctx:
                _, loss = self.model(X, Y)

            loss.backward(create_graph= True)
            g_list = []
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    g_list.append(torch.flatten(param.grad.double()))

            g_tensor = torch.cat(g_list, dim = 0)
            
            self.model.zero_grad(set_to_none = True)
            g_tensor = g_tensor.cuda()
            l = torch.sum(g_tensor*d_tensor)
            l.backward(retain_graph = True)

            hd_list = []
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    hd_list.append(torch.flatten(param.grad.double().data.clone()))

            hd_tensor = torch.cat(hd_list, dim = 0)
            self.model.zero_grad(set_to_none = True)
            hd_tensor = hd_tensor.cpu()
            total_hd_tensor += hd_tensor

            if batch_idx % 10 == 1 or batch_idx == self.gradient_accumulation_steps-1:
                print(f'full hessian: load_iter ={self.ckpt_iteration} current random direction = {v_step} / {self.num_v}, lanczos step = {l_step} / {self.m}, Hd current batch = {batch_idx} / {self.num_batches}, time = {time.time() -t_hd}')
                t_hd = time.time()

            if self.use_minibatch == True and batch_idx == self.gradient_accumulation_steps-1:
                break
        return total_hd_tensor

    def get_batch(self, batch_idx):
        start_idx = batch_idx * self.batch_size * self.block_size
        end_idx = (batch_idx + 1) * self.batch_size * self.block_size
        X = torch.from_numpy((self.train_data[start_idx:end_idx]).astype(np.int64)).reshape(self.batch_size, self.block_size)
        Y = torch.from_numpy((self.train_data[start_idx+1:end_idx+1]).astype(np.int64)).reshape(self.batch_size, self.block_size)
       
        X, Y = X.pin_memory().to(self.device, non_blocking=True), Y.pin_memory().to(self.device, non_blocking=True)

        return X, Y




    def get_true_curve(self, grid, eigenvalues):
        curve = []
        for t in grid:
            density = self.gaussian_density(t, eigenvalues)
            value = np.mean(density)
            curve.append(value)
        return curve
        

    def gaussian_density(self, t, values):
        coeff = 1.0 / np.sqrt(2 * math.pi * self.sigma**2)
        val = -(values - t) ** 2
        val = val / (2.0 * self.sigma**2)
        val = np.exp(val)
        density = coeff * val
        return density






