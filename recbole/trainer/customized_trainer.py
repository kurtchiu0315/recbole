from recbole.trainer import Trainer
from tqdm import tqdm
import torch
import wandb
from time import time
from torch.nn.utils.clip_grad import clip_grad_norm_
from recbole.utils import early_stopping,  dict2str, set_color, get_gpu_usage
from recbole.data.interaction import Interaction
from recbole.model.sequential_recommender.dadaclrec import PolicyChooser as  DADA_PolicyChooser
from recbole.model.sequential_recommender.dndclrec import PolicyChooser as DND_PolicyChooser



class CoSeRecTrainer(Trainer):
    r"""CoSeRecTrainer is designed for CoSeRec, which is a contrastive learning recommendation method based on SASRec.
    """
    def __init__(self, config, model):
        super(CoSeRecTrainer, self).__init__(config, model)
        self.cl_aug_warm_up_epoch = config['cl_aug_warm_up_epoch']
        self.if_hybrid = self.cl_aug_warm_up_epoch != None

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        if self.if_hybrid and epoch_idx >= self.cl_aug_warm_up_epoch:
            item_embedding = self.model.item_embedding.weight[:train_data.item_num]
            train_data.update_embedding_matrix(item_embedding.cpu().detach().numpy())
        return super()._train_epoch(train_data, epoch_idx, loss_func=loss_func, show_progress=show_progress)


class BiCATTrainer(Trainer):
    def __init__(self, config, model):
        super(BiCATTrainer, self).__init__(config, model)
        
        self.num_prior = config['num_prior'] # K
        self.pretrain_epoch = config['pretrain_epoch']
        self.pretrain_only = config['pretrain_only'] or False
        self.show_pretrain_progress = config['show_pretrain_progress'] or False

    def _augment_seq_epoch(self, train_data, epoch_idx= 0, show_progress=False):
        self.model.eval()
        iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                ncols=100,
            ) if show_progress else train_data
        )

        new_gen_seqs = []
        new_gen_seq_lens = []
        labels = []
        for batch_idx, interaction in enumerate(iter_data):
          
            interaction = interaction.to(self.device)
            out = self.model.augment_seq_generator(interaction)
            new_gen_seqs.append(out["new_gen_seq"])
            new_gen_seq_lens.append(out["new_gen_seq_len"])
            labels.append(out["label"])


        return {
            'new_gen_seq': torch.cat(new_gen_seqs, dim = 0),
            'new_gen_seq_len': torch.cat(new_gen_seq_lens, dim = 0),
            'label': torch.cat(labels, dim = 0),
        }


    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):

        if self.pretrain_only:
            self.model.calculate_loss = self.model.prev_item_generator_training
            return super().fit(train_data, valid_data=valid_data, verbose=verbose, saved=saved, show_progress=show_progress, callback_fn=callback_fn)

        else: # do pretraining and fine-tuning
            for epoch_idx in range(self.pretrain_epoch):
                self._train_epoch(train_data, epoch_idx, loss_func=self.model.prev_item_generator_training, show_progress=self.show_pretrain_progress)
            print("End of Pretrain")
            
            out = None
            for i in range(self.num_prior):
                out = self._augment_seq_epoch(train_data)
                train_data.dataset.inter_feat.interaction[self.model.ITEM_SEQ] = out["new_gen_seq"].clone()
                train_data.dataset.inter_feat.interaction[self.model.ITEM_SEQ_LEN] = out["new_gen_seq_len"].clone()
                train_data.dataset.inter_feat.interaction[self.model.POS_ITEM_ID] = out["label"].clone()
            del out
            
            return super().fit(train_data, valid_data=valid_data, verbose=verbose, saved=saved, show_progress=show_progress, callback_fn=callback_fn)


class DSSRecTrainer(Trainer):
   
    def __init__(self, config, model):
        super(DSSRecTrainer, self).__init__(config, model)
        self.pretrain_epoch = config['pretrain_epoch']
        self.pretrain_only = config['pretrain_only'] or False
        self.show_pretrain_progress = config['show_pretrain_progress'] or False


    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        if self.pretrain_only:
            self.model.calculate_loss = self.model.pretrain
            return super().fit(train_data, valid_data=valid_data, verbose=verbose, saved=saved, show_progress=show_progress, callback_fn=callback_fn)

        else: # do pretraining and fine-tuning
            for epoch_idx in range(self.pretrain_epoch):
                self._train_epoch(train_data, epoch_idx, loss_func=self.model.pretrain, show_progress=self.show_pretrain_progress)
            print("End of Pretrain")

            return super().fit(train_data, valid_data=valid_data, verbose=verbose, saved=saved, show_progress=show_progress, callback_fn=callback_fn)



class EC4SRecTrainer(Trainer):
    def __init__(self, config, model):
        super(EC4SRecTrainer, self).__init__(config, model)
        self.start_ecl_epoch = config['start_ecl_epoch']
        self.update_freq = config['update_freq'] or 5

    def _get_item_importances(self, train_data):
        self.model.train()
        loss_func = self.model.calculate_loss
    
        total_importance_scores = 0.0
        iter_data = train_data
        self.optimizer.zero_grad()
        for batch_idx, interaction in enumerate(iter_data):
            
            interaction = interaction.to(self.device)
            loss = loss_func(interaction, get_importance_mode=True)
            self._check_nan(loss)
            loss.backward()
            grad = self.model.item_embedding.weight.grad.detach().clone().abs()
            item_importance_scores = grad.sum(axis=1)/grad.sum()
            with torch.no_grad():
                total_importance_scores += item_importance_scores

        return total_importance_scores/total_importance_scores.sum()
    
    def augmenatation(self, train_data):
        item_importance_scores = self._get_item_importances(train_data)
        aug1s = []
        aug_len1s = []
        aug2s = []
        aug_len2s = []

        for interaction in train_data:
            interaction = train_data.explanation_contrastive_learning_augmentation(
                            interaction, item_importance_scores)
            aug1s.append(interaction['aug1'])
            aug2s.append(interaction['aug2'])
            aug_len1s.append(interaction['aug_len1'])
            aug_len2s.append(interaction['aug_len2'])
        
        train_data.dataset.inter_feat.update(
            Interaction({
                'aug1':torch.cat(aug1s, dim = 0),
                'aug2':torch.cat(aug2s, dim = 0),
                'aug_len1':torch.cat(aug_len1s, dim = 0),
                'aug_len2':torch.cat(aug_len2s, dim = 0),
            })
        )
        return train_data
            

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
      
        if saved and self.start_epoch >= self.epochs:
            self._save_checkpoint(-1)

        self.eval_collector.data_collect(train_data)

        wandb.watch(self.model, log_freq=1)
        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            
            if (epoch_idx >= self.start_ecl_epoch) and (epoch_idx-self.start_ecl_epoch) % self.update_freq == 0:
                self.model.do_CL = True
                start_aug_time = time()
                train_data = self.augmenatation(train_data)
                end_aug_time = time()
                aug_time_output = (set_color("epoch %d augmentation", 'green') + " [" + set_color("time", 'blue')
                                    + ": %.2fs]") % (epoch_idx, end_aug_time-start_aug_time)
                if verbose:
                    self.logger.info(aug_time_output)

            training_start_time = time()
            train_loss = self._train_epoch(train_data, epoch_idx, show_progress=show_progress)
            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            if verbose:
                self.logger.info(train_loss_output)
            self._add_train_loss_to_tensorboard(epoch_idx, train_loss)
            self._add_train_loss_to_wandb(epoch_idx, train_loss)
            wandb.log({'time/train': training_end_time - training_start_time}, step=epoch_idx)

            # eval
            if self.eval_step <= 0 or not valid_data:
                if saved:
                    self._save_checkpoint(epoch_idx)
                    update_output = set_color('Saving current', 'blue') + ': %s' % self.saved_model_file
                    if verbose:
                        self.logger.info(update_output)
                continue
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data, show_progress=show_progress)
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score,
                    self.best_valid_score,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=self.valid_metric_bigger
                )
                valid_end_time = time()
                valid_score_output = (set_color("epoch %d evaluating", 'green') + " [" + set_color("time", 'blue')
                                    + ": %.2fs, " + set_color("valid_score", 'blue') + ": %f]") % \
                                     (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = set_color('valid result', 'blue') + ': \n' + dict2str(valid_result)
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                self.tensorboard.add_scalar('Vaild_score', valid_score, epoch_idx)
                wandb.log({f'val/{k}': v for k, v in valid_result.items()}, step=epoch_idx)
                wandb.log({f'val/best_score': self.best_valid_score,
                           f'time/val': valid_end_time - valid_start_time}, step=epoch_idx)

                if update_flag:
                    if saved:
                        self._save_checkpoint(epoch_idx)
                        update_output = set_color('Saving current best', 'blue') + ': %s' % self.saved_model_file
                        if verbose:
                            self.logger.info(update_output)
                    self.best_valid_result = valid_result

                if callback_fn:
                    callback_fn(epoch_idx, valid_score)

                if stop_flag:
                    stop_output = 'Finished training, best eval result in epoch %d' % \
                                  (epoch_idx - self.cur_step * self.eval_step)
                    if verbose:
                        self.logger.info(stop_output)
                    break
        self._add_hparam_to_tensorboard(self.best_valid_score)
        return self.best_valid_score, self.best_valid_result


class DADACLRecTrainer(Trainer):
    def __init__(self, config, model):
        super(DADACLRecTrainer, self).__init__(config, model)
        self.pc_lr = config['pc_lr']
        self.update_freq = config['update_freq']
        self.hard_sample = config['hard_sample']
        self.pc_epochs = config['pc_epochs']
        self.start_policy_controll_epoch = config['start_policy_controll_epoch']
        self.ad_atk = config['ad_atk']
        self.random_init_update = config['random_init_update']
        self.cl_lambda = config['cl_lambda']
        self.scl_lambda = config['scl_lambda']
        self.gumbel_tau = config['gumbel_tau']
        self.randomness = config['randomness']
        
        self._aug_domain_config(config)

        self.policy_chooser = PolicyChooser(aug_num=self.augnum, config=config).to(next(self.model.parameters()).device)

        self.policy_chooser_optim = torch.optim.Adam(self.policy_chooser.parameters(), lr=self.pc_lr)
        self.training_phase = 'model' # model/ad_atk

        if self.hard_sample and (not config['aug_for_train']):
            # aug_for_train: use aug-data to train the model, must apply gradient descent
            self.policy_chooser_optim.param_groups[0]["lr"] *= -1
        self.init_emb = torch.nn.Embedding.from_pretrained(self.model.item_embedding.weight.clone(),freeze=True)


    def _aug_domain_config(self, config):
        self.aug_domain = config['aug_domain'] 
        assert (self.aug_domain in ['CL4Rec', 'CL4Rec+SU', 'SU+RSU' ,'ALL'])

        if self.aug_domain == 'CL4Rec':
            self.aug_names = ["crop_aug", "mask_aug", "reorder_aug"]
        elif self.aug_domain == 'CL4Rec+SU':
            self.aug_names = ["crop_aug", "mask_aug", "reorder_aug", "su_aug"]
        elif self.aug_domain == 'SU+RSU':
            self.aug_names = ["su_aug", "re_su_aug"]
        elif self.aug_domain == 'ALL':
            self.aug_names = ["crop_aug", "mask_aug", "reorder_aug", "su_aug", "re_su_aug"]

        self.augnum = len(self.aug_names)


    def _choose_aug(self, augmented_data, augmented_data_len, item_embedding, return_choice=False):
        # augmented_data: [B, augnum, SeqMaxLen]
        # augmented_data_len: [B, augnum]
        prob = self.policy_chooser(augmented_data, self.model.item_embedding)
        # prob: [B, aug_num]
        augmented_data_embedding = item_embedding(augmented_data)
        # augmented_data: [B, augnum, SeqMaxLen, hiddensize]

        if self.training_phase == 'model':
            prob_smoother = self.randomness
        else:
            prob_smoother = 10000 # make softmax's distribution similar to uniform distribution
        
        one_hot = torch.nn.functional.gumbel_softmax(prob/prob_smoother, tau=self.gumbel_tau, hard=True)
        transposeData = augmented_data_embedding.permute(2,3,0,1) # [SeqMaxLen, hiddensize,  B, augnum]

        lens = augmented_data_len.gather(dim=1, index=one_hot.argmax(dim=1).reshape(-1, 1)).flatten()
        weightedsumTransposeData = ((transposeData * one_hot).permute(2,3,0,1) ).sum(axis=1) 
        # [B, SeqMaxLen, hiddensize]
        if return_choice:
            return weightedsumTransposeData, lens, one_hot # [B, SeqMaxLen, hiddensize] , [B], [B, augnum]
        else:
            return weightedsumTransposeData, lens # [B, SeqMaxLen, hiddensize] , [B]
    
    
    
    def get_aug(self, interaction, item_embedding, return_choice=False):
        aug_collection = [interaction[name] for name in self.aug_names]  # [B, SeqMaxLen] for every entry in the list
        aug_len_collection = [interaction[name+"_len"] for name in self.aug_names]  # [B] for every entry in the list
        
        augs = torch.cat([aug.unsqueeze(0) for aug in aug_collection],axis = 0).transpose(0,1) # [B, augnum, SeqMaxLen]
        aug_lens = torch.cat([aug_len.unsqueeze(0) for aug_len in aug_len_collection],axis = 0).transpose(0,1) 
        # [B, augnum]
            
        return self._choose_aug(augs, aug_lens, item_embedding, return_choice)




    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                ncols=100,
                desc=set_color(f"Train {epoch_idx:>8}", 'pink'),
            ) if show_progress else train_data
        )
        self.optimizer.zero_grad()
        choiceSum = 0
        for batch_idx, interaction in enumerate(iter_data):

            interaction = interaction.to(self.device)

            augs, aug_lens, choice = self.get_aug(interaction,  self.model.item_embedding, return_choice=True)
            choiceSum += choice.sum(axis=0)
            losses = loss_func(interaction, augs, aug_lens)
            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            self._check_nan(loss)

            if self.training_phase == 'model':
                loss.backward()
            elif self.training_phase == 'ad_atk':
                (-loss).backward()
        
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            if self.training_phase == 'model':
                self.optimizer.step()
                self.optimizer.zero_grad()
                
            elif self.training_phase == 'ad_atk':
                self.policy_chooser_optim.step()
                self.policy_chooser_optim.zero_grad()
       
            
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))
        if self.training_phase == 'model':
            aug_distribution = choiceSum/choiceSum.sum()
            wandb.log({
                f'aug_distribution/aug_{i+1}': aug_distribution[i].item() \
                    for i in range(self.augnum)
            }
            , step=epoch_idx)
        return total_loss
            

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
      
        if saved and self.start_epoch >= self.epochs:
            self._save_checkpoint(-1)

        self.eval_collector.data_collect(train_data)
        
        wandb.watch(self.model, log_freq=1)
        self.policy_chooser_optim.zero_grad()

        for epoch_idx in range(self.start_epoch, self.epochs):
            self.current_epoch = epoch_idx
            if epoch_idx < self.start_policy_controll_epoch:
                self.policy_chooser.apply(self.policy_chooser._init_weights_biasver)

            if epoch_idx >= self.start_policy_controll_epoch and epoch_idx % self.update_freq == 0:
                # Train Policy Chooser
                self.policy_chooser.train()
                self.model.eval()
                if not self.ad_atk:
                    self.policy_chooser_optim.step()
                    self.policy_chooser_optim.zero_grad()
                    if self.random_init_update:
                        self.policy_chooser.apply(self.policy_chooser._init_weights_biasver)
                    if verbose:
                        info = "[" + set_color(f'{epoch_idx} PC update!', 'green') + "]"
                        self.logger.info(info)

                else: # adverserial attack
                    self.training_phase = 'ad_atk'
                    self.policy_chooser_optim.zero_grad()
                    for pc_epoch_index in range(self.pc_epochs):
                        ind = f"ad_atk: {epoch_idx} - {pc_epoch_index}"
                        start_time = time()
                        loss = self._train_epoch(train_data, ind, loss_func=self.model.CL_loss, show_progress=show_progress)  
                        end__time = time()
                        if verbose:
                            info = set_color(f'{ind}', 'green') + ' [' + set_color('time', 'blue') + f': {(end__time-start_time):.2f}s, ' + \
                                set_color("cl_loss", 'blue') + f": {loss:.4f}]"
                            self.logger.info(info)

                self.policy_chooser.eval()
                self.model.train()
                    
                  
            self.training_phase = 'model'
            training_start_time = time()
            wandb.log({'lr/policy_chooser': self.policy_chooser_optim.param_groups[0]["lr"]}, step=epoch_idx)
            wandb.log({'lr/main': self.optimizer.param_groups[0]["lr"]}, step=epoch_idx)
 
            train_loss = self._train_epoch(train_data, epoch_idx, show_progress=show_progress)

            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            if verbose:
                self.logger.info(train_loss_output)
            self._add_train_loss_to_tensorboard(epoch_idx, train_loss)
            self._add_train_loss_to_wandb(epoch_idx, train_loss)
            wandb.log({'time/train': training_end_time - training_start_time}, step=epoch_idx)

            # eval
            if self.eval_step <= 0 or not valid_data:
                if saved:
                    self._save_checkpoint(epoch_idx)
                    update_output = set_color('Saving current', 'blue') + ': %s' % self.saved_model_file
                    if verbose:
                        self.logger.info(update_output)
                continue
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data, show_progress=show_progress)
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score,
                    self.best_valid_score,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=self.valid_metric_bigger
                )

                valid_end_time = time()
                valid_score_output = (set_color("epoch %d evaluating", 'green') + " [" + set_color("time", 'blue')
                                    + ": %.2fs, " + set_color("valid_score", 'blue') + ": %f]") % \
                                     (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = set_color('valid result', 'blue') + ': \n' + dict2str(valid_result)
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                self.tensorboard.add_scalar('Vaild_score', valid_score, epoch_idx)
                wandb.log({f'val/{k}': v for k, v in valid_result.items()}, step=epoch_idx)
                wandb.log({f'val/best_score': self.best_valid_score,
                           f'time/val': valid_end_time - valid_start_time}, step=epoch_idx)

                if update_flag:
                    if saved:
                        self._save_checkpoint(epoch_idx)
                        update_output = set_color('Saving current best', 'blue') + ': %s' % self.saved_model_file
                        if verbose:
                            self.logger.info(update_output)
                    self.best_valid_result = valid_result

                if callback_fn:
                    callback_fn(epoch_idx, valid_score)

                if stop_flag:
                    stop_output = 'Finished training, best eval result in epoch %d' % \
                                  (epoch_idx - self.cur_step * self.eval_step)
                    if verbose:
                        self.logger.info(stop_output)
                    break
        self._add_hparam_to_tensorboard(self.best_valid_score)
        return self.best_valid_score, self.best_valid_result



class DNDCLRecTrainer(Trainer):
    def __init__(self, config, model):
        super(DNDCLRecTrainer, self).__init__(config, model)
        self.pc_lr = config['pc_lr']
        self.update_freq = config['update_freq']
        self.pc_epochs = config['pc_epochs']
        self.start_policy_controll_epoch = config['start_policy_controll_epoch']
        self.gumbel_tau = config['gumbel_tau']
        self.randomness = config['randomness']
        
        self._aug_domain_config(config)

        self.policy_chooser = DND_PolicyChooser(aug_num=self.augnum, config=config).to(next(self.model.parameters()).device)

        self.policy_chooser_optim = torch.optim.Adam(self.policy_chooser.parameters(), lr=self.pc_lr)
        self.training_phase = 'model' # model/policy_update


    def _aug_domain_config(self, config):
        self.aug_domain = config['aug_domain'] 
        assert (self.aug_domain in ['CL4Rec', 'CL4Rec+SU', 'SU+RSU' ,'ALL', 'RSU', 'SU'])

        if self.aug_domain == 'CL4Rec':
            self.aug_names = ["crop_aug", "mask_aug", "reorder_aug"]
        elif self.aug_domain == 'CL4Rec+SU':
            self.aug_names = ["crop_aug", "mask_aug", "reorder_aug", "su_aug"]
        elif self.aug_domain == 'SU+RSU':
            self.aug_names = ["su_aug", "re_su_aug"]
        elif self.aug_domain == 'ALL':
            self.aug_names = ["crop_aug", "mask_aug", "reorder_aug", "su_aug", "re_su_aug"]
        elif self.aug_domain == 'RSU':
            self.aug_names = ["re_su_aug"]
        elif self.aug_domain == 'SU':
            self.aug_names = ["su_aug"]

        self.augnum = len(self.aug_names)


    def _choose_aug(self, augmented_data, augmented_data_len, item_embedding, return_choice=False):
        # augmented_data: [B, augnum, SeqMaxLen]
        # augmented_data_len: [B, augnum]
        prob = self.policy_chooser(augmented_data, self.model.item_embedding)
        # TODO: supervised learning probability
        
        # prob: [B, aug_num]
        augmented_data_embedding = item_embedding(augmented_data)
        # augmented_data_embedding: [B, augnum, SeqMaxLen, hiddensize]

        if self.training_phase == 'model':
            prob_smoother = self.randomness
        else:
            prob_smoother = 10000 # make softmax's distribution similar to uniform distribution
        
        one_hot = torch.nn.functional.gumbel_softmax(prob/prob_smoother, tau=self.gumbel_tau, hard=True)
        transposeData = augmented_data_embedding.permute(2,3,0,1) # [SeqMaxLen, hiddensize,  B, augnum]

        lens = augmented_data_len.gather(dim=1, index=one_hot.argmax(dim=1).reshape(-1, 1)).flatten()
        weightedsumTransposeData = ((transposeData * one_hot).permute(2,3,0,1) ).sum(axis=1) 
        # [B, SeqMaxLen, hiddensize]
        if return_choice:
            return weightedsumTransposeData, lens, one_hot # [B, SeqMaxLen, hiddensize] , [B], [B, augnum]
        else:
            return weightedsumTransposeData, lens # [B, SeqMaxLen, hiddensize] , [B]
    
    
    
    def get_aug(self, interaction, item_embedding, return_choice=False):
        aug_collection = [interaction[name] for name in self.aug_names]  # [B, SeqMaxLen] for every entry in the list
        aug_len_collection = [interaction[name+"_len"] for name in self.aug_names]  # [B] for every entry in the list
        
        augs = torch.cat([aug.unsqueeze(0) for aug in aug_collection],axis = 0).transpose(0,1) # [B, augnum, SeqMaxLen]
        aug_lens = torch.cat([aug_len.unsqueeze(0) for aug_len in aug_len_collection],axis = 0).transpose(0,1) 
        # [B, augnum]
            
        return self._choose_aug(augs, aug_lens, item_embedding, return_choice)


    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                ncols=100,
                desc=set_color(f"Train {epoch_idx:>8}", 'pink'),
            ) if show_progress else train_data
        )
        self.optimizer.zero_grad()
        choiceSum = 0
        new_policy_probs = None
        for batch_idx, interaction in enumerate(iter_data):

            interaction = interaction.to(self.device)
            # interaction update

            augs, aug_lens, choice = self.get_aug(interaction,  self.model.item_embedding, return_choice=True)
            # print(choice)

            choiceSum += choice.sum(axis=0)
            if self.training_phase == 'model':
                losses = loss_func(interaction, augs, aug_lens)
            elif self.training_phase == 'policy_update':
                losses, performance = loss_func(interaction, augs, aug_lens, reward_mode=True)
                # print(interaction['policy_prob'])
                # new_policy_prob = (10*(interaction['policy_prob'] + choice * performance.unsqueeze(0).T)).softmax(dim=1)

                # if isinstance(new_policy_probs, type(None)):
                #     new_policy_probs = new_policy_prob
                # else:
                #     new_policy_probs = torch.vstack([new_policy_probs, new_policy_prob])
                # supervised_pc_loss = self.policy_chooser.calculate_loss(choice, new_policy_prob.detach().clone())
                # print(losses)
                # print(supervised_pc_loss)
                # losses = losses + tuple([supervised_pc_loss])

            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            self._check_nan(loss)
            loss.backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)

            if self.training_phase == 'model':
                self.optimizer.step()
                self.optimizer.zero_grad()
            elif self.training_phase == 'policy_update':
                self.policy_chooser_optim.step()
                self.policy_chooser_optim.zero_grad()
       
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))
        # if self.training_phase == 'policy_update':
        #     # train_data.dataset.inter_feat.interaction["performance"] = performances
        #     train_data.dataset.inter_feat.interaction['policy_prob'] = new_policy_probs.to("cpu")
        if self.training_phase == 'model':
            aug_distribution = choiceSum/choiceSum.sum()
            wandb.log({
                f'aug_distribution/aug_{i+1}': aug_distribution[i].item() \
                    for i in range(self.augnum)
            }
            , step=epoch_idx)
        return total_loss
            

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
      
        if saved and self.start_epoch >= self.epochs:
            self._save_checkpoint(-1)

        self.eval_collector.data_collect(train_data)
        train_data.dataset.inter_feat.interaction['policy_prob'] = torch.ones(len(train_data.dataset), self.augnum).softmax(dim=1)
        wandb.watch(self.model, log_freq=1)
        self.policy_chooser_optim.zero_grad()

        for epoch_idx in range(self.start_epoch, self.epochs):
            self.current_epoch = epoch_idx
            if epoch_idx < self.start_policy_controll_epoch: 
                self.policy_chooser.apply(self.policy_chooser._init_weights_biasver)

            if epoch_idx >= self.start_policy_controll_epoch \
                and epoch_idx % self.update_freq == 0 \
                and self.cur_step < self.update_freq:
                # Train Policy Chooser
                self.policy_chooser.train()
                self.model.eval()  
         
                self.training_phase = 'policy_update'
                self.policy_chooser_optim.zero_grad()
                for pc_epoch_index in range(self.pc_epochs):
                    ind = f"policy_update: {epoch_idx} - {pc_epoch_index}"
                    start_time = time()
                    loss = self._train_epoch(train_data, ind, show_progress=show_progress)  
                    end__time = time()
                    if verbose:
                        info = set_color(f'{ind}', 'green') + ' [' + set_color('time', 'blue') + f': {(end__time-start_time):.2f}s, ' + \
                            set_color("last_layer_sim", 'blue') + f": {loss[0]:.4f} | " + \
                            set_color("first_layer_sim", 'blue') + f": {-loss[1]:.4f}]"
                        self.logger.info(info)
                    

                self.policy_chooser.eval()
                self.model.train()
                    
                  
            self.training_phase = 'model'
            training_start_time = time()
            wandb.log({'lr/policy_chooser': self.policy_chooser_optim.param_groups[0]["lr"]}, step=epoch_idx)
            wandb.log({'lr/main': self.optimizer.param_groups[0]["lr"]}, step=epoch_idx)
 
            train_loss = self._train_epoch(train_data, epoch_idx, show_progress=show_progress)

            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            if verbose:
                self.logger.info(train_loss_output)
            self._add_train_loss_to_tensorboard(epoch_idx, train_loss)
            self._add_train_loss_to_wandb(epoch_idx, train_loss)
            wandb.log({'time/train': training_end_time - training_start_time}, step=epoch_idx)

            # eval
            if self.eval_step <= 0 or not valid_data:
                if saved:
                    self._save_checkpoint(epoch_idx)
                    update_output = set_color('Saving current', 'blue') + ': %s' % self.saved_model_file
                    if verbose:
                        self.logger.info(update_output)
                continue
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data, show_progress=show_progress)
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score,
                    self.best_valid_score,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=self.valid_metric_bigger
                )

                valid_end_time = time()
                valid_score_output = (set_color("epoch %d evaluating", 'green') + " [" + set_color("time", 'blue')
                                    + ": %.2fs, " + set_color("valid_score", 'blue') + ": %f]") % \
                                     (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = set_color('valid result', 'blue') + ': \n' + dict2str(valid_result)
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                self.tensorboard.add_scalar('Vaild_score', valid_score, epoch_idx)
                wandb.log({f'val/{k}': v for k, v in valid_result.items()}, step=epoch_idx)
                wandb.log({f'val/best_score': self.best_valid_score,
                           f'time/val': valid_end_time - valid_start_time}, step=epoch_idx)

                if update_flag:
                    if saved:
                        self._save_checkpoint(epoch_idx)
                        update_output = set_color('Saving current best', 'blue') + ': %s' % self.saved_model_file
                        if verbose:
                            self.logger.info(update_output)
                    self.best_valid_result = valid_result

                if callback_fn:
                    callback_fn(epoch_idx, valid_score)

                if stop_flag:
                    stop_output = 'Finished training, best eval result in epoch %d' % \
                                  (epoch_idx - self.cur_step * self.eval_step)
                    if verbose:
                        self.logger.info(stop_output)
                    break
        self._add_hparam_to_tensorboard(self.best_valid_score)
        return self.best_valid_score, self.best_valid_result

class MCLRecTrainer(Trainer):
    r"""RecVAETrainer is designed for MCLRec, which is a sequential recommender.

    """

    def __init__(self, config, model):
        super(MCLRecTrainer, self).__init__(config, model)
        self.joint=config['joint']
        # Extractor
        self.aug_1 = Extractor([64, 32,16,64], "gelu", None).to(self.device)
        self.aug_2 = Extractor([64, 32,16,64], "gelu", None).to(self.device)
        # optimize two different extractors
        self.optimizer_1 = self._build_optimizer([{"params": self.aug_1.parameters()}, {"params": self.aug_2.parameters()}])
        # joint-learing
        self.optimizer_2 = self._build_optimizer([{"params": self.model.parameters()},{"params": self.aug_1.parameters()}, {"params": self.aug_2.parameters()}])


    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        r"""Train the model in an epoch

        Args:
            train_data (DataLoader): The train data.
            epoch_idx (int): The current epoch id.
            loss_func (function): The loss function of :attr:`model`. If it is ``None``, the loss function will be
                :attr:`self.model.calculate_loss`. Defaults to ``None``.
            show_progress (bool): Show the progress of training epoch. Defaults to ``False``.

        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, it will return a
            tuple which includes the sum of loss in each part.
        """
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        iter_data = (
            tqdm(
                enumerate(train_data),
                total=len(train_data),
                desc=set_color(f"Train {epoch_idx:>5}", 'pink'),
            ) if show_progress else enumerate(train_data)
        )
        for batch_idx, interaction in iter_data:
            interaction = interaction.to(self.device)
            if self.joint==1:
                self.optimizer_2.zero_grad()
                losses = loss_func(interaction,[self.aug_1, self.aug_2])
                if isinstance(losses, tuple):
                    loss = sum(losses)
                    loss_tuple = tuple(per_loss.item() for per_loss in losses)
                    total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
                else:
                    loss = losses
                    total_loss = losses.item() if total_loss is None else total_loss + losses.item()
                self._check_nan(loss)
                loss.backward()
                if self.clip_grad_norm:
                    clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
                self.optimizer_2.step()
            else:
                # step 1, update the parameters of the encoder
                self.optimizer.zero_grad()
                for param in self.aug_1.parameters():
                    param.requires_grad = False
                for param in self.aug_2.parameters():
                    param.requires_grad = False
                losses = loss_func(interaction, [self.aug_1, self.aug_2], "step1")
                if isinstance(losses, tuple):
                    loss = sum(losses)
                    loss_tuple = tuple(per_loss.item() for per_loss in losses)
                    total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
                else:
                    loss = losses
                    total_loss = losses.item() if total_loss is None else total_loss + losses.item()
                self._check_nan(loss)
                loss.backward()
                if self.clip_grad_norm:
                    clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
                self.optimizer.step()

                # step 2，update the parameters of two learnable extractors
                self.optimizer_1.zero_grad()
                for param in self.aug_1.parameters():
                    param.requires_grad = True
                for param in self.aug_2.parameters():
                    param.requires_grad = True
                losses = loss_func(interaction, [self.aug_1, self.aug_2], "step2")
                if isinstance(losses, tuple):
                    loss = sum(losses)
                    loss_tuple = tuple(per_loss.item() for per_loss in losses)
                    total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
                else:
                    loss = losses
                    total_loss = losses.item() if total_loss is None else total_loss + losses.item()
                self._check_nan(loss)
                loss.backward()
                if self.clip_grad_norm:
                    clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
                self.optimizer_1.step()

                # step 3， update the parameters of the encoder
                self.optimizer.zero_grad()
                for param in self.aug_1.parameters():
                    param.requires_grad = False
                for param in self.aug_2.parameters():
                    param.requires_grad = False
                losses = loss_func(interaction, [self.aug_1, self.aug_2], "step3")
                if isinstance(losses, tuple):
                    loss = sum(losses)
                    loss_tuple = tuple(per_loss.item() for per_loss in losses)
                    total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
                else:
                    loss = losses
                    total_loss = losses.item() if total_loss is None else total_loss + losses.item()
                self._check_nan(loss)
                loss.backward()
                if self.clip_grad_norm:
                    clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
                self.optimizer.step()

        return total_loss
