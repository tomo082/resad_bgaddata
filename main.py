import os
import warnings
import argparse
from tqdm import tqdm
import numpy as np
import torch
import timm
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train import train
from validate import validate
from datasets.mvtec import MVTEC, MVTECANO
from datasets.visa import VISA, VISAANO
from datasets.btad import BTAD
from datasets.mvtec_3d import MVTEC3D
from datasets.mpdd import MPDD
from datasets.mvtec_loco import MVTECLOCO
from datasets.brats import BRATS

from models.fc_flow import load_flow_model
from models.modules import MultiScaleConv
from models.vq import MultiScaleVQ
from utils import init_seeds, get_residual_features, get_mc_matched_ref_features, get_mc_reference_features
from utils import BoundaryAverager
from losses.loss import calculate_log_barrier_bi_occ_loss
from classes import VISA_TO_MVTEC, MVTEC_TO_VISA, MVTEC_TO_BTAD, MVTEC_TO_MVTEC3D
from classes import MVTEC_TO_MPDD, MVTEC_TO_MVTECLOCO, MVTEC_TO_BRATS, MVTEC_TO_MVTEC

warnings.filterwarnings('ignore')

TOTAL_SHOT = 4  # total few-shot reference samples
FIRST_STAGE_EPOCH = 10
SETTINGS = {'visa_to_mvtec': VISA_TO_MVTEC, 'mvtec_to_visa': MVTEC_TO_VISA,
            'mvtec_to_btad': MVTEC_TO_BTAD, 'mvtec_to_mvtec3d': MVTEC_TO_MVTEC3D,
            'mvtec_to_mpdd': MVTEC_TO_MPDD, 'mvtec_to_mvtecloco': MVTEC_TO_MVTECLOCO,
            'mvtec_to_brats': MVTEC_TO_BRATS,'mvtec_to_mvtec': MVTEC_TO_MVTEC}


def main(args):
    if args.setting in SETTINGS.keys():
        CLASSES = SETTINGS[args.setting]
    else:
        raise ValueError(f"Dataset setting must be in {SETTINGS.keys()}, but got {args.setting}.")
    
    if CLASSES['seen'][0] in MVTEC.CLASS_NAMES:  # from mvtec to other datasets
        train_dataset1 = MVTEC(args.train_dataset_dir, class_name=CLASSES['seen'], train=True, 
                               normalize="w50",
                               img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
        train_loader1 = DataLoader(
            train_dataset1, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True
        )
        train_dataset2 = MVTECANO(args.train_dataset_dir, class_name=CLASSES['seen'], train=True, 
                                  normalize='w50',
                                  img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
        train_loader2 = DataLoader(
            train_dataset2, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True
        )
    else:  # from visa to mvtec
        train_dataset1 = VISA(args.train_dataset_dir, class_name=CLASSES['seen'], train=True,
                               normalize="w50",
                               img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
        train_loader1 = DataLoader(
            train_dataset1, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True
        )
        train_dataset2 = VISAANO(args.train_dataset_dir, class_name=CLASSES['seen'], train=True, 
                                 normalize="w50",
                                 img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
        train_loader2 = DataLoader(
            train_dataset2, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True
        )
    
    encoder = timm.create_model('wide_resnet50_2', features_only=True, 
            out_indices=(1, 2, 3), pretrained=True).eval()  # the pretrained checkpoint will be in /home/.cache/torch/hub/checkpoints/
    encoder = encoder.to(args.device)
    feat_dims = encoder.feature_info.channels()
    
    boundary_ops = BoundaryAverager(num_levels=args.feature_levels)
    vq_ops = MultiScaleVQ(num_embeddings=args.num_embeddings, channels=feat_dims).to(args.device)
    optimizer_vq = torch.optim.Adam(vq_ops.parameters(), lr=args.lr, weight_decay=0.0005)
    scheduler_vq = torch.optim.lr_scheduler.MultiStepLR(optimizer_vq, milestones=[70, 90], gamma=0.1)
    
    constraintor = MultiScaleConv(feat_dims).to(args.device)
    # weight_decay is the l2 weight penalty lambda, weight_decay = lambda / 2
    optimizer0 = torch.optim.Adam(constraintor.parameters(), lr=args.lr, weight_decay=0.0005)
    scheduler0 = torch.optim.lr_scheduler.MultiStepLR(optimizer0, milestones=[70, 90], gamma=0.1)
    
    # Normflow decoder
    estimators = [load_flow_model(args, feat_dim) for feat_dim in feat_dims]
    estimators = [decoder.to(args.device) for decoder in estimators]
    params = list(estimators[0].parameters())
    for l in range(1, args.feature_levels):
        params += list(estimators[l].parameters())
    optimizer1 = torch.optim.Adam(params, lr=args.lr, weight_decay=0.0005)
    scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer1, milestones=[70, 90], gamma=0.1)
    
    #best_pro = 0　#-1になってしまうので変更
    best_img_auc = 0
    N_batch = 8192
            
    # 最良モデルのエポックで保存するためのデータ保持用
    best_epoch_class_data = {}   
            
    for epoch in range(args.epochs):
        vq_ops.train()
        constraintor.train()
        for estimator in estimators:
            estimator.train()
        if epoch < FIRST_STAGE_EPOCH:
            train_loader = train_loader1
        else:
            train_loader = train_loader2
        train_loss_total, total_num = 0, 0
        progress_bar = tqdm(total=len(train_loader))
        progress_bar.set_description(f"Epoch[{epoch}/{args.epochs}]")
#data aug を適応するならここ? supervisedだからあまり意味ない?
        for step, batch in enumerate(train_loader):
            progress_bar.update(1)
            images, _, masks, class_names, anomaly_types = batch
            
            images = images.to(args.device)
            masks = masks.to(args.device)
            
            with torch.no_grad():
                features = encoder(images)
            
            ref_features = get_mc_reference_features(encoder, args.train_dataset_dir, class_names, images.device, args.train_ref_shot)
            mfeatures = get_mc_matched_ref_features(features, class_names, ref_features)
            rfeatures = get_residual_features(features, mfeatures, pos_flag=True)
            
            lvl_masks = []
            for l in range(args.feature_levels):
                _, _, h, w = rfeatures[l].size()
                m = F.interpolate(masks, size=(h, w), mode='nearest').squeeze(1)
                lvl_masks.append(m)
            rfeatures_t = [rfeature.detach().clone() for rfeature in rfeatures]
            
            loss_vq = vq_ops(rfeatures, lvl_masks, train=True)
            train_loss_total += loss_vq.item()
            total_num += 1
            optimizer_vq.zero_grad()
            loss_vq.backward()
            optimizer_vq.step()
            
            rfeatures = constraintor(*rfeatures)
            loss = 0
            for l in range(args.feature_levels):  # backward svdd loss
                e = rfeatures[l]  
                t = rfeatures_t[l]
                bs, dim, h, w = e.size()
                e = e.permute(0, 2, 3, 1).reshape(-1, dim)
                t = t.permute(0, 2, 3, 1).reshape(-1, dim)
                m = lvl_masks[l]
                m = m.reshape(-1)
                
                loss_i, _, _ = calculate_log_barrier_bi_occ_loss(e, m, t)
                loss += loss_i
            optimizer0.zero_grad()
            loss.backward()
            optimizer0.step()
            
            train_loss_total += loss.item()
            total_num += 1
            
            # detach the rfeatures for flow optimization
            rfeatures = [rfeature.detach().clone() for rfeature in rfeatures]
            # train flow corresponding to with neck
            loss, num = train(args, rfeatures, estimators, optimizer1, masks, boundary_ops, epoch, N_batch=N_batch, FIRST_STAGE_EPOCH=FIRST_STAGE_EPOCH)
            train_loss_total += loss
            total_num += num
        
        scheduler_vq.step()
        scheduler0.step()
        scheduler1.step()
               
        progress_bar.close()
        print(f"Epoch[{epoch}/{args.epochs}]: train_loss: {train_loss_total / total_num}")
        
        if (epoch + 1) % args.eval_freq == 0:
            s1_res, s2_res, s_res = [], [], []
            test_ref_features = load_mc_reference_features(args.test_ref_feature_dir, CLASSES['unseen'], args.device, args.num_ref_shot)
            # 各クラスの評価結果とデータを一時的に保持する辞書
            current_epoch_class_data_for_saving = {}
                    
            for class_name_eval in CLASSES['unseen']:
                if class_name_eval in MVTEC.CLASS_NAMES:
                    test_dataset = MVTEC(args.test_dataset_dir, class_name=class_name_eval, train=False,
                                         normalize='w50',
                                         img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
                elif class_name_eval in VISA.CLASS_NAMES:
                    test_dataset = VISA(args.test_dataset_dir, class_name=class_name_eval, train=False,
                                        normalize='w50',
                                        img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
                elif class_name_eval in BTAD.CLASS_NAMES:
                    test_dataset = BTAD(args.test_dataset_dir, class_name=class_name_eval, train=False,
                                        normalize='w50',
                                        img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
                elif class_name_eval in MVTEC3D.CLASS_NAMES:
                    test_dataset = MVTEC3D(args.test_dataset_dir, class_name=class_name_eval, train=False,
                                           normalize='w50',
                                           img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
                elif class_name_eval in MPDD.CLASS_NAMES:
                    test_dataset = MPDD(args.test_dataset_dir, class_name=class_name_eval, train=False,
                                        normalize='w50',
                                        img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
                elif class_name_eval in MVTECLOCO.CLASS_NAMES:
                    test_dataset = MVTECLOCO(args.test_dataset_dir, class_name=class_name_eval, train=False,
                                        normalize='w50',
                                        img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
                elif class_name_eval in BRATS.CLASS_NAMES:
                    test_dataset = BRATS(args.test_dataset_dir, class_name=class_name_eval, train=False,
                                           normalize='w50',
                                           img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
                else:
                    raise ValueError('Unrecognized class name: {}'.format(class_name_eval))
                test_loader = DataLoader(
                    test_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False
                )
                #metrics = validate(args, encoder, vq_ops, constraintor, estimators, test_loader, test_ref_features[class_name], args.device, class_name)
                metrics = validate(args, encoder, vq_ops, constraintor, estimators, test_loader, 
                                   test_ref_features[class_name_eval], args.device, class_name_eval)
                
                img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro = metrics['scores']
                
                print("Epoch: {}, Class Name: {}, Image AUC | AP | F1_Score: {} | {} | {}, Pixel AUC | AP | F1_Score | AUPRO: {} | {} | {} | {}".format(
                    epoch, class_name_eval, img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro))
                s1_res.append(metrics['scores1'])
                s2_res.append(metrics['scores2'])
                s_res.append(metrics['scores'])
                # 各クラスの評価結果から特徴量とラベルデータを一時的に保存
                current_epoch_class_data_for_saving[class_name_eval] = {
                    'features': metrics['features'],
                    'anomaly_types': metrics['anomaly_types'],
                    'gts_labels': metrics['gts_labels']
                }            
            s1_res = np.array(s1_res)
            s2_res = np.array(s2_res)
            s_res = np.array(s_res)
            img_auc1, img_ap1, img_f1_score1, pix_auc1, pix_ap1, pix_f1_score1, pix_aupro1 = np.mean(s1_res, axis=0)
            img_auc2, img_ap2, img_f1_score2, pix_auc2, pix_ap2, pix_f1_score2, pix_aupro2 = np.mean(s2_res, axis=0)
            img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro = np.mean(s_res, axis=0)
            print('(Logps) Average Image AUC | AP | F1_Score: {:.3f} | {:.3f} | {:.3f}, Average Pixel AUC | AP | F1_Score | AUPRO: {:.3f} | {:.3f} | {:.3f} | {:.3f}'.format(
                img_auc1, img_ap1, img_f1_score1, pix_auc1, pix_ap1, pix_f1_score1, pix_aupro1))
            print('(BScores) Average Image AUC | AP | F1_Score: {:.3f} | {:.3f} | {:.3f}, Average Pixel AUC | AP | F1_Score | AUPRO: {:.3f} | {:.3f} | {:.3f} | {:.3f}'.format(
                img_auc2, img_ap2, img_f1_score2, pix_auc2, pix_ap2, pix_f1_score2, pix_aupro2))
            print('(Merged) Average Image AUC | AP | F1_Score: {:.3f} | {:.3f} | {:.3f}, Average Pixel AUC | AP | F1_Score | AUPRO: {:.3f} | {:.3f} | {:.3f} | {:.3f}'.format(
                img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro))
            
            if img_auc > best_img_auc: #pix_aupro > best_pro:
                os.makedirs(args.checkpoint_path, exist_ok=True)
                best_img_auc = img_auc #best_pro = pix_aupro
                state_dict = {'vq_ops': vq_ops.state_dict(),
                              'constraintor': constraintor.state_dict(),
                              'estimators': [estimator.state_dict() for estimator in estimators]}
                torch.save(state_dict, os.path.join(args.checkpoint_path, f'{args.setting}_checkpoints.pth'))
                # 新しい特徴量保存ディレクトリを作成
                features_save_dir = os.path.join(args.checkpoint_path, 'features_for_analysis')
                os.makedirs(features_save_dir, exist_ok=True) 
                for class_name_to_save, data in current_epoch_class_data_for_saving.items():
                    # クラス名の下にファイルを保存するパスを構築
                    class_specific_save_dir = os.path.join(features_save_dir, class_name_to_save)
                    os.makedirs(class_specific_save_dir, exist_ok=True)

                    features_filename = os.path.join(class_specific_save_dir, f'epoch{epoch}_features.npy')
                    anomaly_types_filename = os.path.join(class_specific_save_dir, f'epoch{epoch}_anomaly_types.npy')
                    gts_labels_filename = os.path.join(class_specific_save_dir, f'epoch{epoch}_gts_labels.npy')

                    np.save(features_filename, data['features'])
                    np.save(anomaly_types_filename, data['anomaly_types'])
                    np.save(gts_labels_filename, data['gts_labels'])
                    print(f"  - クラス '{class_name_to_save}': Epoch {epoch} のデータを {class_specific_save_dir} に保存しました。")
                
                # 最良エポックのデータなので、今後の可視化のためにこれを覚えておく
                best_epoch_class_data = current_epoch_class_data_for_saving.copy()


def load_mc_reference_features(root_dir: str, class_names, device: torch.device, num_shot=4):
    refs = {}
    for class_name in class_names:
        layer1_refs = np.load(os.path.join(root_dir, class_name, 'layer1.npy'))
        layer2_refs = np.load(os.path.join(root_dir, class_name, 'layer2.npy'))
        layer3_refs = np.load(os.path.join(root_dir, class_name, 'layer3.npy'))
        
        layer1_refs = torch.from_numpy(layer1_refs).to(device)
        layer2_refs = torch.from_numpy(layer2_refs).to(device)
        layer3_refs = torch.from_numpy(layer3_refs).to(device)
        
        K1 = (layer1_refs.shape[0] // TOTAL_SHOT) * num_shot
        layer1_refs = layer1_refs[:K1, :]
        K2 = (layer2_refs.shape[0] // TOTAL_SHOT) * num_shot
        layer2_refs = layer2_refs[:K2, :]
        K3 = (layer3_refs.shape[0] // TOTAL_SHOT) * num_shot
        layer3_refs = layer3_refs[:K3, :]
        
        refs[class_name] = (layer1_refs, layer2_refs, layer3_refs)
    
    return refs
                    
                    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting', type=str, default="visa_to_mvtec")
    parser.add_argument('--train_dataset_dir', type=str, default="")
    parser.add_argument('--test_dataset_dir', type=str, default="")
    parser.add_argument('--test_ref_feature_dir', type=str, default="./ref_features/w50/mvtec_4shot")
    
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--checkpoint_path', type=str, default="./checkpoints/")
    parser.add_argument('--eval_freq', type=int, default=1)
    parser.add_argument('--backbone', type=str, default="wide_resnet50_2")
    
    # flow parameters
    parser.add_argument('--flow_arch', type=str, default='conditional_flow_model')
    parser.add_argument('--feature_levels', default=3, type=int)
    parser.add_argument('--coupling_layers', type=int, default=10)
    parser.add_argument('--clamp_alpha', type=float, default=1.9)
    parser.add_argument('--pos_embed_dim', type=int, default=256)
    parser.add_argument('--pos_beta', type=float, default=0.05)
    parser.add_argument('--margin_tau', type=float, default=0.1)
    parser.add_argument('--bgspp_lambda', type=float, default=1)
    
    parser.add_argument('--fdm_alpha', type=float, default=0.4)  # low value, more training distribution
    parser.add_argument('--num_embeddings', type=int, default=1536)  # VQ embeddings
    parser.add_argument("--train_ref_shot", type=int, default=4)
    parser.add_argument("--num_ref_shot", type=int, default=4)
    
    args = parser.parse_args()
    init_seeds(42)
    
    main(args)
    

    
    
            
